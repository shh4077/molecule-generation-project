"""Prepare Janus Block_bAE training data from train/test CSV files.

The upstream Janus training script expects:

    data_dir/
      vocabulary.json
      sequences.h5

where sequences.h5 has:

    tokens:  flat int token stream
    indices: [start, length] rows into tokens

This script converts SMILES CSV files into that format while preserving a
manifest that records how many rows were accepted, skipped, or truncated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical_smile",
    "molecule_smiles",
    "SMILES",
)


@dataclass
class Counters:
    rows_seen: int = 0
    accepted: int = 0
    invalid_smiles: int = 0
    selfies_errors: int = 0
    unknown_token: int = 0
    truncated: int = 0
    duplicate: int = 0
    empty: int = 0


def infer_smiles_column(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    lowered = {c.lower(): c for c in columns}
    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        f"Could not infer SMILES column for {path}. "
        f"Available columns: {columns}. Pass --smiles-column explicitly."
    )


def iter_smiles_from_csv(
    paths: Iterable[Path],
    smiles_column: str | None,
    chunksize: int,
    limit: int | None,
) -> Iterator[tuple[str, str, int]]:
    yielded = 0
    for path in paths:
        column = infer_smiles_column(path, smiles_column)
        for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[column]):
            for local_idx, value in enumerate(chunk[column].tolist()):
                if limit is not None and yielded >= limit:
                    return
                yielded += 1
                yield str(value) if not pd.isna(value) else "", path.name, local_idx


def canonicalize_smiles(smiles: str, keep_stereo: bool) -> str | None:
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)


def load_vocab(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    if "stoi" not in vocab and "itos_lst" in vocab:
        vocab["stoi"] = {token: i for i, token in enumerate(vocab["itos_lst"])}
    if "itos_lst" not in vocab and "stoi" in vocab:
        vocab["itos_lst"] = [None] * len(vocab["stoi"])
        for token, idx in vocab["stoi"].items():
            vocab["itos_lst"][idx] = token
    return vocab


def build_vocab_from_csv(
    paths: list[Path],
    smiles_column: str | None,
    chunksize: int,
    limit: int | None,
    keep_stereo: bool,
) -> dict:
    token_set: set[str] = set()
    for smiles, _source, _row in tqdm(
        iter_smiles_from_csv(paths, smiles_column, chunksize, limit),
        desc="Building SELFIES vocab",
    ):
        canonical = canonicalize_smiles(smiles, keep_stereo)
        if canonical is None:
            continue
        try:
            token_set.update(sf.split_selfies(sf.encoder(canonical)))
        except Exception:
            continue
    itos = SPECIAL_TOKENS + sorted(token_set)
    return {"stoi": {token: i for i, token in enumerate(itos)}, "itos_lst": itos}


def append_1d(dataset, values: np.ndarray) -> int:
    start = int(dataset.shape[0])
    dataset.resize((start + len(values),))
    dataset[start:] = values
    return start


def append_2d(dataset, values: np.ndarray) -> None:
    start = int(dataset.shape[0])
    dataset.resize((start + len(values), dataset.shape[1]))
    dataset[start:] = values


def append_strings(dataset, values: list[str]) -> None:
    start = int(dataset.shape[0])
    dataset.resize((start + len(values),))
    dataset[start:] = values


def encode_one(
    raw_smiles: str,
    token_to_id: dict[str, int],
    max_len: int,
    keep_stereo: bool,
    allow_unk: bool,
) -> tuple[list[int] | None, str | None, str | None]:
    canonical = canonicalize_smiles(raw_smiles, keep_stereo)
    if canonical is None:
        return None, None, "invalid_smiles"

    try:
        tokens = list(sf.split_selfies(sf.encoder(canonical)))
    except Exception:
        return None, canonical, "selfies_errors"

    sos_id = token_to_id["<sos>"]
    eos_id = token_to_id["<eos>"]
    unk_id = token_to_id["<unk>"]

    truncated = False
    if len(tokens) > max_len - 2:
        tokens = tokens[: max_len - 2]
        truncated = True

    token_ids = [sos_id]
    for token in tokens:
        idx = token_to_id.get(token)
        if idx is None:
            if not allow_unk:
                return None, canonical, "unknown_token"
            idx = unk_id
        token_ids.append(idx)
    token_ids.append(eos_id)
    return token_ids, canonical, "truncated" if truncated else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert train/test SMILES CSV files into Janus Block_bAE HDF5 data."
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--vocab-mode", choices=["paper", "build"], default="paper")
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compression", choices=["none", "gzip"], default="none")
    parser.add_argument("--keep-stereo", action="store_true")
    parser.add_argument("--allow-unk", action="store_true")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--no-smiles-txt", action="store_true")
    args = parser.parse_args()

    paths = [args.train_csv]
    if args.test_csv:
        paths.append(args.test_csv)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.vocab_path is None:
        args.vocab_path = (
            Path(__file__).resolve().parents[1]
            / "external"
            / "Janus-QUBO"
            / "Block_bAE"
            / "vocab"
            / "vocabulary.json"
        )

    if args.vocab_mode == "build":
        vocab = build_vocab_from_csv(
            paths, args.smiles_column, args.chunksize, args.limit, args.keep_stereo
        )
        with (args.out_dir / "vocabulary.json").open("w", encoding="utf-8") as handle:
            json.dump(vocab, handle, indent=2)
    else:
        vocab = load_vocab(args.vocab_path)
        shutil.copyfile(args.vocab_path, args.out_dir / "vocabulary.json")

    token_to_id = vocab["stoi"]
    counters = Counters()
    seen: set[str] = set()

    compression = None if args.compression == "none" else "gzip"
    h5_path = args.out_dir / "sequences.h5"
    smiles_txt_path = args.out_dir / "clean_smiles.txt"
    smiles_txt = None if args.no_smiles_txt else smiles_txt_path.open("w", encoding="utf-8")

    try:
        with h5py.File(h5_path, "w") as h5:
            tokens_ds = h5.create_dataset(
                "tokens",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int16,
                chunks=(1_000_000,),
                compression=compression,
            )
            indices_ds = h5.create_dataset(
                "indices",
                shape=(0, 2),
                maxshape=(None, 2),
                dtype=np.int64,
                chunks=(100_000, 2),
                compression=compression,
            )
            str_dtype = h5py.string_dtype(encoding="utf-8")
            smiles_ds = h5.create_dataset(
                "smiles",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dtype,
                chunks=(100_000,),
                compression=compression,
            )

            index_buffer: list[tuple[int, int]] = []
            smiles_buffer: list[str] = []

            iterator = iter_smiles_from_csv(
                paths, args.smiles_column, args.chunksize, args.limit
            )
            for raw_smiles, _source, _row in tqdm(iterator, desc="Encoding SMILES"):
                counters.rows_seen += 1
                if not raw_smiles.strip():
                    counters.empty += 1
                    continue

                token_ids, canonical, status = encode_one(
                    raw_smiles,
                    token_to_id,
                    args.max_len,
                    args.keep_stereo,
                    args.allow_unk,
                )
                if token_ids is None:
                    if status:
                        setattr(counters, status, getattr(counters, status) + 1)
                    continue

                if args.dedupe and canonical in seen:
                    counters.duplicate += 1
                    continue
                seen.add(canonical)

                if status == "truncated":
                    counters.truncated += 1

                start = append_1d(tokens_ds, np.asarray(token_ids, dtype=np.int16))
                index_buffer.append((start, len(token_ids)))
                smiles_buffer.append(canonical or "")
                counters.accepted += 1

                if smiles_txt is not None:
                    smiles_txt.write((canonical or "") + "\n")

                if len(index_buffer) >= 100_000:
                    append_2d(indices_ds, np.asarray(index_buffer, dtype=np.int64))
                    append_strings(smiles_ds, smiles_buffer)
                    index_buffer.clear()
                    smiles_buffer.clear()

            if index_buffer:
                append_2d(indices_ds, np.asarray(index_buffer, dtype=np.int64))
                append_strings(smiles_ds, smiles_buffer)

            h5.attrs["max_len"] = args.max_len
            h5.attrs["vocab_size"] = len(vocab["itos_lst"])
    finally:
        if smiles_txt is not None:
            smiles_txt.close()

    manifest = {
        "inputs": [str(path) for path in paths],
        "h5_path": str(h5_path),
        "vocabulary_path": str(args.out_dir / "vocabulary.json"),
        "clean_smiles_txt": None if args.no_smiles_txt else str(smiles_txt_path),
        "vocab_mode": args.vocab_mode,
        "max_len": args.max_len,
        "dedupe": args.dedupe,
        "allow_unk": args.allow_unk,
        "keep_stereo": args.keep_stereo,
        "counters": asdict(counters),
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest["counters"], indent=2))
    print(f"Wrote {h5_path}")


if __name__ == "__main__":
    main()

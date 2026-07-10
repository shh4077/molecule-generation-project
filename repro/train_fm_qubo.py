"""Train a small FM surrogate on Janus latent codes and export a QUBO matrix.

This is intentionally standalone. The upstream Janus FM code is useful, but it
is logger/GPU/DDP oriented and has no direct CLI for a local HDF5 latent file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import RDConfig
    import sys

    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    from sascorer import calculateScore as calculate_sa_score
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("RDKit SA_Score/sascorer is required for this script") from exc


class TorchFM(torch.nn.Module):
    def __init__(self, n_features: int, factor_k: int):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(n_features, factor_k) * 0.01)
        self.lin = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x, self.V.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        return (out_inter + self.lin(x)).squeeze(1)


def read_smiles_from_h5(h5_path: Path) -> list[str] | None:
    with h5py.File(h5_path, "r") as h5:
        if "smiles" not in h5:
            return None
        raw = h5["smiles"][:]
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def ro5_pass(mol) -> bool:
    return (
        Descriptors.MolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
    )


def compute_properties(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for i, smi in enumerate(tqdm(smiles, desc="RDKit properties")):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append(
                {
                    "row": i,
                    "smiles": smi,
                    "valid": False,
                    "qed": np.nan,
                    "sa": np.nan,
                    "logp": np.nan,
                    "logp_in_1_5": False,
                    "ro5": False,
                }
            )
            continue
        logp = float(Crippen.MolLogP(mol))
        rows.append(
            {
                "row": i,
                "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
                "valid": True,
                "qed": float(QED.qed(mol)),
                "sa": float(calculate_sa_score(mol)),
                "logp": logp,
                "logp_in_1_5": 1.0 <= logp <= 5.0,
                "ro5": ro5_pass(mol),
            }
        )
    return pd.DataFrame(rows)


def interval_score(values: np.ndarray, low: float = 1.0, high: float = 5.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    below = np.maximum(low - values, 0.0)
    above = np.maximum(values - high, 0.0)
    distance = below + above
    return 1.0 / (1.0 + distance)


def make_target(props: pd.DataFrame, target: str) -> np.ndarray:
    if target == "qed":
        y = props["qed"].to_numpy(np.float32)
    elif target == "neg_sa":
        y = -props["sa"].to_numpy(np.float32)
    elif target == "logp_window":
        y = interval_score(props["logp"].to_numpy(np.float32))
    elif target == "ro5":
        y = props["ro5"].astype(np.float32).to_numpy()
    elif target == "composite":
        qed = props["qed"].to_numpy(np.float32)
        sa = props["sa"].to_numpy(np.float32)
        sa_good = 1.0 - np.clip((sa - 1.0) / 9.0, 0.0, 1.0)
        logp_good = interval_score(props["logp"].to_numpy(np.float32))
        ro5 = props["ro5"].astype(np.float32).to_numpy()
        y = 0.40 * qed + 0.25 * sa_good + 0.20 * logp_good + 0.15 * ro5
    else:
        raise ValueError(f"Unknown target: {target}")
    return y.astype(np.float32)


def load_external_labels(label_csv: Path, smiles_column: str, target_column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in pd.read_csv(label_csv, chunksize=100_000, usecols=[smiles_column, target_column]):
        for smi, score in zip(chunk[smiles_column], chunk[target_column]):
            if pd.isna(smi) or pd.isna(score):
                continue
            canonical = canonicalize(str(smi))
            if canonical is None or canonical in out:
                continue
            out[canonical] = float(score)
    return out


def train_val_split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    return idx[n_val:], idx[:n_val]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y_true))}


def save_qubo(model: TorchFM, output_csv: Path) -> None:
    V = model.V.detach().cpu().numpy()
    w = model.lin.weight.detach().cpu().numpy().squeeze()
    n = V.shape[0]
    qubo = np.zeros((n, n), dtype=np.float64)
    interactions = V @ V.T
    for i in range(n):
        qubo[i, i] = -w[i]
        for j in range(i + 1, n):
            qubo[i, j] = -interactions[i, j]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, qubo, delimiter=",", fmt="%.8f")


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FM surrogate and export QUBO CSV.")
    parser.add_argument("--latent-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smiles-file", type=Path)
    parser.add_argument("--label-csv", type=Path)
    parser.add_argument("--label-smiles-column", default="smiles")
    parser.add_argument("--target-column")
    parser.add_argument(
        "--target",
        choices=["composite", "qed", "neg_sa", "logp_window", "ro5"],
        default="composite",
    )
    parser.add_argument("--factor-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.latent_h5, "r") as h5:
        X = h5["latent_codes"][:]
    X = X[:, :128].astype(np.float32)

    smiles = read_smiles_from_h5(args.latent_h5)
    if smiles is None and args.smiles_file:
        smiles = [line.strip() for line in args.smiles_file.read_text(encoding="utf-8").splitlines()]
    if smiles is None:
        raise ValueError("No smiles dataset found in H5. Pass --smiles-file.")

    if args.limit:
        X = X[: args.limit]
        smiles = smiles[: args.limit]

    props = compute_properties(smiles)
    valid_mask = props["valid"].to_numpy(bool)

    if args.label_csv and args.target_column:
        labels = load_external_labels(
            args.label_csv, args.label_smiles_column, args.target_column
        )
        aligned = np.array([labels.get(smi, np.nan) for smi in props["smiles"]], dtype=np.float32)
        valid_mask &= np.isfinite(aligned)
        y = aligned
        target_description = f"{args.label_csv}:{args.target_column}"
    else:
        y = make_target(props, args.target)
        target_description = args.target

    X = X[valid_mask]
    y = y[valid_mask]
    props = props.loc[valid_mask].reset_index(drop=True)

    if len(X) < 100:
        raise ValueError(f"Too few valid training rows after filtering: {len(X)}")

    y_mean = float(np.mean(y))
    y_std = float(np.std(y) or 1.0)
    y_scaled = (y - y_mean) / y_std

    train_idx, val_idx = train_val_split(len(X), args.val_fraction, args.seed)
    model = TorchFM(n_features=X.shape[1], factor_k=args.factor_k).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y_scaled[train_idx])
    X_val = torch.from_numpy(X[val_idx]).to(args.device)
    y_val = y_scaled[val_idx]

    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(X_train))
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            xb = X_train[batch_idx].to(args.device)
            yb = y_train[batch_idx].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val).detach().cpu().numpy()
        m = metrics(y_val, pred_val)
        m["epoch"] = epoch
        m["train_loss"] = float(np.mean(losses))
        history.append(m)
        print(
            f"epoch={epoch:03d} train_loss={m['train_loss']:.5f} "
            f"val_r2={m['r2']:.4f} val_mae={m['mae']:.4f}"
        )

    model_path = args.output_dir / "fm_surrogate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": X.shape[1],
            "factor_k": args.factor_k,
            "target_mean": y_mean,
            "target_std": y_std,
            "target_description": target_description,
            "args": jsonable_args(args),
        },
        model_path,
    )
    qubo_csv = args.output_dir / "qubo_128.csv"
    save_qubo(model, qubo_csv)

    props["target"] = y
    props.to_csv(args.output_dir / "training_properties.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "fm_training_history.csv", index=False)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "latent_h5": str(args.latent_h5),
                "n_rows": int(len(X)),
                "target": target_description,
                "target_mean": y_mean,
                "target_std": y_std,
                "model_path": str(model_path),
                "qubo_csv": str(qubo_csv),
                "last_epoch": history[-1],
                "args": jsonable_args(args),
            },
            handle,
            indent=2,
        )

    print(f"Wrote {model_path}")
    print(f"Wrote {qubo_csv}")


if __name__ == "__main__":
    main()

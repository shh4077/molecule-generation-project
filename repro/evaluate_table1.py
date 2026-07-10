"""Evaluate generated SMILES using Janus-QUBO Table 1 style metrics.

The script supports four ranking policies:

- input: preserve the order supplied by the generator.
- score: rank once by an explicit generator score/energy column and evaluate all
  properties on the same top 5/10/20% subsets. This is the recommended mode
  when the sampler writes QUBO energy or model score.
- metric: rank separately by each observed property. This is an oracle-style
  upper-bound analysis and should not be confused with generator ranking.
- composite: rank once by a transparent heuristic composite score.

CSV input may contain additional columns such as ``energy`` or ``score``.
HDF5 input must contain ``smiles`` and may additionally contain ``energies``,
``energy``, ``scores``, or ``score``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import RDConfig

    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    from sascorer import calculateScore as calculate_sa_score
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "RDKit SA_Score/sascorer is required. Ensure RDKit Contrib is installed."
    ) from exc


PAPER_TABLE_1: dict[str, dict[str, list[float]]] = {
    "MolGPT": {
        "QED": [0.76, 0.74, 0.70, 0.56],
        "SA": [3.71, 3.83, 3.98, 4.80],
        "LogP": [87.40, 85.83, 81.41, 69.63],
        "Ro5": [96.08, 94.85, 91.04, 69.13],
    },
    "JTVAE": {
        "QED": [0.77, 0.74, 0.71, 0.55],
        "SA": [3.53, 3.73, 3.88, 4.43],
        "LogP": [85.56, 81.02, 74.62, 62.44],
        "Ro5": [88.89, 83.32, 77.90, 61.79],
    },
    "Janus-BER": {
        "QED": [0.54, 0.54, 0.52, 0.39],
        "SA": [3.31, 3.66, 4.04, 5.24],
        "LogP": [37.21, 39.54, 42.46, 44.48],
        "Ro5": [99.90, 99.81, 99.56, 96.13],
    },
    "Janus-GA": {
        "QED": [0.70, 0.69, 0.68, 0.60],
        "SA": [2.97, 3.22, 3.50, 4.57],
        "LogP": [87.27, 86.16, 84.65, 77.96],
        "Ro5": [93.32, 91.88, 89.98, 85.84],
    },
    "Janus-RBM": {
        "QED": [0.64, 0.63, 0.61, 0.54],
        "SA": [2.79, 3.02, 3.29, 4.38],
        "LogP": [69.24, 68.13, 66.88, 60.96],
        "Ro5": [99.92, 99.85, 99.74, 98.91],
    },
    "Janus-FM": {
        "QED": [0.80, 0.79, 0.76, 0.65],
        "SA": [2.99, 3.21, 3.48, 4.57],
        "LogP": [98.30, 97.70, 96.69, 91.68],
        "Ro5": [99.62, 99.56, 99.53, 99.39],
    },
}

SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "SMILES",
    "canonical_smiles",
    "molecule_smiles",
)
RANK_COLUMN_CANDIDATES = (
    "energy",
    "energies",
    "qubo_energy",
    "objective",
    "score",
    "scores",
    "predicted_score",
    "prediction",
)


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def infer_smiles_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested SMILES column {requested!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    lowered = {str(column).lower(): str(column) for column in df.columns}
    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    raise ValueError(
        "Could not infer a SMILES column. "
        f"Available columns: {list(df.columns)}. Pass --smiles-column explicitly."
    )


def read_candidates(path: Path, smiles_column: str | None) -> pd.DataFrame:
    """Read candidate SMILES while preserving generator scores when available."""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h5:
            if "smiles" not in h5:
                raise ValueError(f"HDF5 file {path} does not contain a 'smiles' dataset.")
            data: dict[str, Any] = {"smiles": _decode_h5_strings(h5["smiles"][:])}
            n_rows = len(data["smiles"])
            for name in RANK_COLUMN_CANDIDATES:
                if name in h5 and h5[name].ndim == 1 and len(h5[name]) == n_rows:
                    data[name] = np.asarray(h5[name][:])
        df = pd.DataFrame(data)

    elif suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
        column = infer_smiles_column(df, smiles_column)
        if column != "smiles":
            df = df.rename(columns={column: "smiles"})

    else:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        df = pd.DataFrame({"smiles": lines})

    if "smiles" not in df.columns:
        raise ValueError(f"No SMILES column was found in {path}.")

    df = df.copy()
    if "input_order" in df.columns:
        df = df.rename(columns={"input_order": "source_input_order"})
    df["smiles"] = df["smiles"].where(df["smiles"].notna(), "").astype(str)
    df.insert(0, "input_order", np.arange(len(df), dtype=np.int64))
    return df


def ro5_pass(mol: Chem.Mol) -> bool:
    """Return whether a molecule satisfies the standard Lipinski Rule of Five."""
    return bool(
        Descriptors.MolWt(mol) <= 500.0
        and Crippen.MolLogP(mol) <= 5.0
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
    )


def logp_distance_to_window(logp: float, low: float, high: float) -> float:
    if logp < low:
        return low - logp
    if logp > high:
        return logp - high
    return 0.0


def compute_properties(
    candidates: pd.DataFrame,
    *,
    logp_low: float,
    logp_high: float,
    keep_stereo: bool,
) -> pd.DataFrame:
    """Compute RDKit properties while preserving input metadata."""
    rows: list[dict[str, Any]] = []

    for record in tqdm(
        candidates.to_dict(orient="records"),
        desc="Evaluating SMILES",
        total=len(candidates),
    ):
        smi = str(record.get("smiles", ""))
        out = dict(record)
        out["original_smiles"] = smi

        mol = Chem.MolFromSmiles(smi) if smi and smi.lower() != "fail" else None
        if mol is None:
            out.update(
                {
                    "canonical_smiles": None,
                    "valid": False,
                    "qed": np.nan,
                    "sa": np.nan,
                    "logp": np.nan,
                    "logp_in_window": False,
                    "logp_distance": np.nan,
                    "ro5": False,
                }
            )
            rows.append(out)
            continue

        try:
            canonical = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=keep_stereo,
            )
            logp = float(Crippen.MolLogP(mol))
            out.update(
                {
                    "canonical_smiles": canonical,
                    "valid": True,
                    "qed": float(QED.qed(mol)),
                    "sa": float(calculate_sa_score(mol)),
                    "logp": logp,
                    "logp_in_window": bool(logp_low <= logp <= logp_high),
                    "logp_distance": float(
                        logp_distance_to_window(logp, logp_low, logp_high)
                    ),
                    "ro5": ro5_pass(mol),
                }
            )
        except Exception:
            out.update(
                {
                    "canonical_smiles": None,
                    "valid": False,
                    "qed": np.nan,
                    "sa": np.nan,
                    "logp": np.nan,
                    "logp_in_window": False,
                    "logp_distance": np.nan,
                    "ro5": False,
                }
            )
        rows.append(out)

    return pd.DataFrame(rows)


def infer_rank_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested rank column {requested!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    for candidate in RANK_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Ranking mode 'score' requires --rank-column or one of these columns: "
        + ", ".join(RANK_COLUMN_CANDIDATES)
    )


def infer_rank_ascending(column: str, requested_order: str) -> bool:
    if requested_order == "asc":
        return True
    if requested_order == "desc":
        return False

    lowered = column.lower()
    if any(token in lowered for token in ("energy", "loss", "distance")):
        return True
    return False


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent heuristic score used only when explicitly requested."""
    qed = df["qed"].to_numpy(dtype=np.float64)
    sa_good = 1.0 - np.clip((df["sa"].to_numpy(dtype=np.float64) - 1.0) / 9.0, 0.0, 1.0)
    logp_good = 1.0 / (1.0 + df["logp_distance"].to_numpy(dtype=np.float64))
    ro5 = df["ro5"].astype(float).to_numpy()
    score = qed + sa_good + logp_good + ro5
    return df.assign(_composite_score=score)


def rank_once(
    df: pd.DataFrame,
    *,
    ranking: str,
    rank_column: str | None,
    rank_order: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one common ranked list for input, score, or composite modes."""
    if ranking == "input":
        return (
            df.sort_values("input_order", kind="mergesort").reset_index(drop=True),
            {"rank_column": "input_order", "ascending": True},
        )

    if ranking == "score":
        column = infer_rank_column(df, rank_column)
        ascending = infer_rank_ascending(column, rank_order)
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().all():
            raise ValueError(f"Rank column {column!r} contains no numeric values.")
        ranked = (
            df.assign(_rank_value=numeric)
            .dropna(subset=["_rank_value"])
            .sort_values("_rank_value", ascending=ascending, kind="mergesort")
            .reset_index(drop=True)
        )
        return ranked, {"rank_column": column, "ascending": ascending}

    if ranking == "composite":
        ranked = (
            add_composite_score(df)
            .sort_values("_composite_score", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        return ranked, {"rank_column": "_composite_score", "ascending": False}

    raise ValueError(f"rank_once does not support ranking={ranking!r}")


def metric_ranked_subset(df: pd.DataFrame, prop: str, n: int) -> pd.DataFrame:
    """Oracle-style property-specific ranking used only for --ranking metric."""
    if prop == "QED":
        return df.nlargest(n, "qed", keep="first")
    if prop == "SA":
        return df.nsmallest(n, "sa", keep="first")
    if prop == "LogP":
        return df.nsmallest(n, "logp_distance", keep="first")
    if prop == "Ro5":
        # Ro5 itself is binary. QED is used only as a deterministic tie-breaker.
        return (
            df.assign(_ro5_int=df["ro5"].astype(int))
            .sort_values(
                ["_ro5_int", "qed", "input_order"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .head(n)
        )
    raise ValueError(prop)


def summarize_table1(
    df_valid: pd.DataFrame,
    *,
    ranking: str,
    rank_column: str | None,
    rank_order: str,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Compute top 5/10/20% and full-set metrics."""
    if df_valid.empty:
        raise ValueError("No valid generated SMILES were found.")

    common_ranked: pd.DataFrame | None = None
    ranking_metadata: dict[str, Any] = {"mode": ranking}
    if ranking != "metric":
        common_ranked, rank_meta = rank_once(
            df_valid,
            ranking=ranking,
            rank_column=rank_column,
            rank_order=rank_order,
        )
        ranking_metadata.update(rank_meta)
        if common_ranked.empty:
            raise ValueError("No candidates remained after ranking/filtering.")

    out: dict[str, list[float]] = {}
    for prop in ("QED", "SA", "LogP", "Ro5"):
        base = df_valid if common_ranked is None else common_ranked
        values: list[float] = []

        for fraction in (0.05, 0.10, 0.20):
            n = max(1, int(np.ceil(len(base) * fraction)))
            if ranking == "metric":
                subset = metric_ranked_subset(df_valid, prop, n)
            else:
                subset = base.head(n)

            if prop == "QED":
                values.append(float(subset["qed"].mean()))
            elif prop == "SA":
                values.append(float(subset["sa"].mean()))
            elif prop == "LogP":
                values.append(float(subset["logp_in_window"].mean() * 100.0))
            else:
                values.append(float(subset["ro5"].mean() * 100.0))

        # The Avg column is always computed on the complete valid population.
        if prop == "QED":
            values.append(float(df_valid["qed"].mean()))
        elif prop == "SA":
            values.append(float(df_valid["sa"].mean()))
        elif prop == "LogP":
            values.append(float(df_valid["logp_in_window"].mean() * 100.0))
        else:
            values.append(float(df_valid["ro5"].mean() * 100.0))

        out[prop] = values

    return out, ranking_metadata


def observed_to_dataframe(
    model_name: str,
    observed: dict[str, list[float]],
) -> pd.DataFrame:
    labels = ("top_5", "top_10", "top_20", "avg")
    rows: list[dict[str, Any]] = []
    for prop, values in observed.items():
        for label, value in zip(labels, values, strict=True):
            rows.append(
                {
                    "model": model_name,
                    "property": prop,
                    "subset": label,
                    "observed": value,
                }
            )
    return pd.DataFrame(rows)


def flatten_comparison(
    model_name: str,
    observed: dict[str, list[float]],
    paper_class: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = ("top_5", "top_10", "top_20", "avg")
    paper = PAPER_TABLE_1[paper_class]

    for prop, values in observed.items():
        for label, value, ref in zip(labels, values, paper[prop], strict=True):
            abs_delta = abs(value - ref)
            relative_error = abs_delta / max(abs(ref), 1e-12) * 100.0
            rows.append(
                {
                    "model": model_name,
                    "compare_to": paper_class,
                    "property": prop,
                    "subset": label,
                    "observed": value,
                    "paper": ref,
                    "delta": value - ref,
                    "abs_delta": abs_delta,
                    "relative_error_percent": relative_error,
                }
            )
    return pd.DataFrame(rows)


def property_error_summary(comparison: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for prop, group in comparison.groupby("property", sort=False):
        summary[str(prop)] = {
            "mean_abs_delta": float(group["abs_delta"].mean()),
            "mean_relative_error_percent": float(
                group["relative_error_percent"].mean()
            ),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Janus-QUBO Table 1 style metrics."
    )
    parser.add_argument("--smiles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smiles-column")
    parser.add_argument("--model-name", default="My-Janus")
    parser.add_argument(
        "--paper-class",
        choices=sorted(PAPER_TABLE_1),
        default="Janus-FM",
    )
    parser.add_argument(
        "--ranking",
        choices=["input", "score", "metric", "composite"],
        default="input",
        help=(
            "input: preserve generator order; score: sort once by --rank-column; "
            "metric: property-specific oracle ranking; composite: heuristic ranking."
        ),
    )
    parser.add_argument(
        "--rank-column",
        help="Numeric generator score/energy column used with --ranking score.",
    )
    parser.add_argument(
        "--rank-order",
        choices=["auto", "asc", "desc"],
        default="auto",
        help="Sort direction for --ranking score. Auto uses ascending for energy/loss.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Evaluate only unique canonical SMILES, preserving the best-ranked duplicate.",
    )
    parser.add_argument("--keep-stereo", action="store_true")
    parser.add_argument("--logp-low", type=float, default=1.0)
    parser.add_argument("--logp-high", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.logp_low > args.logp_high:
        raise ValueError("--logp-low must not exceed --logp-high.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(args.smiles, args.smiles_column)
    props = compute_properties(
        candidates,
        logp_low=args.logp_low,
        logp_high=args.logp_high,
        keep_stereo=args.keep_stereo,
    )
    props.to_csv(args.output_dir / "generated_properties.csv", index=False)

    total = len(props)
    valid_population = props[props["valid"]].copy().reset_index(drop=True)
    valid_count = len(valid_population)
    unique_valid_count = int(valid_population["canonical_smiles"].nunique())

    if valid_population.empty:
        raise ValueError("No valid generated SMILES were found.")

    # Rank before deduplication so that the highest-ranked occurrence is retained.
    if args.deduplicate:
        if args.ranking == "metric":
            # Property-specific ranking has no single canonical order. Preserve input order.
            valid_population = (
                valid_population.sort_values("input_order", kind="mergesort")
                .drop_duplicates("canonical_smiles", keep="first")
                .reset_index(drop=True)
            )
        else:
            ranked_for_dedupe, _ = rank_once(
                valid_population,
                ranking=args.ranking,
                rank_column=args.rank_column,
                rank_order=args.rank_order,
            )
            valid_population = (
                ranked_for_dedupe.drop_duplicates("canonical_smiles", keep="first")
                .reset_index(drop=True)
            )

    observed, ranking_metadata = summarize_table1(
        valid_population,
        ranking=args.ranking,
        rank_column=args.rank_column,
        rank_order=args.rank_order,
    )

    observed_df = observed_to_dataframe(args.model_name, observed)
    observed_df.to_csv(args.output_dir / "observed_table1.csv", index=False)

    comparison = flatten_comparison(args.model_name, observed, args.paper_class)
    comparison.to_csv(args.output_dir / "table1_comparison.csv", index=False)

    evaluated_count = len(valid_population)
    summary = {
        "input_smiles": str(args.smiles),
        "model_name": args.model_name,
        "paper_class": args.paper_class,
        "ranking": ranking_metadata,
        "deduplicate": bool(args.deduplicate),
        "keep_stereo": bool(args.keep_stereo),
        "logp_window": [args.logp_low, args.logp_high],
        "total": total,
        "valid": valid_count,
        "validity_percent": float(valid_count / max(total, 1) * 100.0),
        "unique_valid": unique_valid_count,
        "unique_valid_percent_of_valid": float(
            unique_valid_count / max(valid_count, 1) * 100.0
        ),
        "evaluated_population": evaluated_count,
        "observed_table1": observed,
        "error_by_property": property_error_summary(comparison),
        "mean_relative_error_percent_all_metrics": float(
            comparison["relative_error_percent"].mean()
        ),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(comparison.to_string(index=False))
    print(f"Wrote {args.output_dir / 'generated_properties.csv'}")
    print(f"Wrote {args.output_dir / 'observed_table1.csv'}")
    print(f"Wrote {args.output_dir / 'table1_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

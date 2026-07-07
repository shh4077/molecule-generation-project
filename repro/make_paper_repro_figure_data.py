#!/usr/bin/env python3
"""Build Figure 2/3/4-style data and plots for Janus-QUBO reproduction.

This script consumes the Experiment A directory produced by
``repro/run_experiment_a.py`` and writes paper-vs-reproduction summary tables
plus figure-ready CSV/PNG artifacts.

The emphasis is on comparison data, not exact visual cloning of the paper:

- Figure 2-style: QED/SA landscape, Pareto front, Table 1 paper-vs-repro.
- Figure 3-style: latent bit contributions and pair interaction summaries.
- Figure 4-style: property/substructure shifts induced by important bits.

LightGBM is used when available for Figure 3 model diagnostics. If it is not
installed, the script still produces empirical bit and pair contribution data.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


PAPER_JANUS_FM: dict[str, dict[str, float]] = {
    "QED": {"5": 0.80, "10": 0.79, "20": 0.76, "Avg": 0.65},
    "SA": {"5": 2.99, "10": 3.21, "20": 3.48, "Avg": 4.57},
    "LogP": {"5": 98.30, "10": 97.70, "20": 96.69, "Avg": 91.68},
    "Ro5": {"5": 99.62, "10": 99.56, "20": 99.53, "Avg": 99.39},
}

DEFAULT_PROTOCOL_DIR = "evaluation_paper_protocol"
DEFAULT_TARGETS = [
    "official_property_contribution",
    "official_final_score",
    "qed",
    "score_sa",
    "score_logp",
    "score_ro5",
]

SMARTS_PATTERNS: dict[str, str] = {
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;!$(N=*);!$(N-[!#6])]",
    "amide": "[CX3](=[OX1])[NX3]",
    "halogen": "[F,Cl,Br,I]",
    "sulfur": "[#16]",
    "aromatic_ring": "a1aaaaa1",
    "hetero_aromatic": "[a;!#6]",
    "carbonyl": "[CX3]=[OX1]",
}


@dataclass(frozen=True)
class SeedData:
    seed: str
    properties: pd.DataFrame
    latent_codes: pd.DataFrame | None


def normalize_subset(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"avg", "average", "all"}:
        return "Avg"
    for prefix in ("top_", "top", "Top", "Top_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("%", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_property(value: object) -> str:
    mapping = {
        "qed": "QED",
        "sa": "SA",
        "sa_score": "SA",
        "logp": "LogP",
        "ro5": "Ro5",
        "lipinski": "Ro5",
    }
    text = str(value).strip()
    return mapping.get(text.lower(), text)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_seed_dirs(experiment_root: Path, seeds: list[str] | None = None) -> list[Path]:
    seed_dirs = [
        p
        for p in experiment_root.iterdir()
        if p.is_dir() and p.name.startswith("seed_") and (p / "final_10000.csv").exists()
    ]
    if seeds:
        wanted = {f"seed_{s}" if not str(s).startswith("seed_") else str(s) for s in seeds}
        seed_dirs = [p for p in seed_dirs if p.name in wanted]
    return sorted(seed_dirs, key=lambda p: p.name)


def load_properties(seed_dir: Path, protocol_dir: str) -> pd.DataFrame:
    candidates = [
        seed_dir / protocol_dir / "generated_properties.csv",
        seed_dir / protocol_dir / "evaluated_population.csv",
        seed_dir / "final_10000.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed_dir.name.replace("seed_", "")
            return df
    raise FileNotFoundError(f"No property CSV found under {seed_dir}")


def decode_latent_hex(hex_value: object, dim: int = 128) -> np.ndarray | None:
    if pd.isna(hex_value):
        return None
    text = str(hex_value).strip()
    if not text:
        return None
    if set(text) <= {"0", "1"} and len(text) >= dim:
        return np.fromiter((int(ch) for ch in text[:dim]), dtype=np.uint8, count=dim)
    if text.startswith("0x"):
        text = text[2:]
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        return None
    if not raw:
        return None
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    if len(bits) < dim:
        return None
    return bits[:dim].astype(np.uint8)


def load_latent_codes(seed_dir: Path, dim: int) -> pd.DataFrame | None:
    """Return one latent row per latent_code_hex when available."""
    rows: list[pd.DataFrame] = []
    for h5_path in sorted((seed_dir / "sampling_rounds").glob("round_*/decoded.h5")):
        try:
            with h5py.File(h5_path, "r") as h5:
                if "latent_codes" not in h5 or "latent_code_hex" not in h5:
                    continue
                codes = np.asarray(h5["latent_codes"][:], dtype=np.uint8)
                hexes = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in h5["latent_code_hex"][:]
                ]
                if codes.ndim != 2:
                    continue
                n_bits = min(codes.shape[1], dim)
                data = pd.DataFrame(codes[:, :n_bits], columns=[f"bit_{i}" for i in range(n_bits)])
                data.insert(0, "latent_code_hex", hexes)
                rows.append(data)
        except OSError as exc:
            warnings.warn(f"Could not read {h5_path}: {exc}")
    if rows:
        out = pd.concat(rows, ignore_index=True)
        return out.drop_duplicates("latent_code_hex")
    return None


def attach_latents(properties: pd.DataFrame, latent_table: pd.DataFrame | None, dim: int) -> pd.DataFrame:
    bit_cols = [f"bit_{i}" for i in range(dim)]
    if latent_table is not None and "latent_code_hex" in properties.columns:
        merged = properties.merge(latent_table, on="latent_code_hex", how="left")
        if merged[bit_cols].notna().all(axis=1).mean() > 0.95:
            for col in bit_cols:
                merged[col] = merged[col].astype(np.uint8)
            return merged

    if "latent_code_hex" not in properties.columns:
        return properties

    decoded = [decode_latent_hex(x, dim=dim) for x in properties["latent_code_hex"]]
    valid = [x is not None for x in decoded]
    if not any(valid):
        return properties
    bits = np.full((len(decoded), dim), np.nan)
    for i, arr in enumerate(decoded):
        if arr is not None:
            bits[i, :] = arr
    bit_df = pd.DataFrame(bits, columns=bit_cols)
    merged = pd.concat([properties.reset_index(drop=True), bit_df], axis=1)
    return merged


def load_seed_data(experiment_root: Path, protocol_dir: str, seeds: list[str] | None, dim: int) -> list[SeedData]:
    out: list[SeedData] = []
    for seed_dir in find_seed_dirs(experiment_root, seeds):
        props = load_properties(seed_dir, protocol_dir)
        latents = load_latent_codes(seed_dir, dim)
        props = attach_latents(props, latents, dim)
        out.append(SeedData(seed=seed_dir.name.replace("seed_", ""), properties=props, latent_codes=latents))
    if not out:
        raise FileNotFoundError(f"No seed directories found under {experiment_root}")
    return out


def pareto_front_qed_sa(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[["qed", "sa"]].dropna().copy()
    if clean.empty:
        return clean
    clean = clean.sort_values(["qed", "sa"], ascending=[False, True]).reset_index(drop=True)
    rows: list[pd.Series] = []
    best_sa = math.inf
    for _, row in clean.iterrows():
        sa = float(row["sa"])
        if sa < best_sa:
            rows.append(row)
            best_sa = sa
    if not rows:
        return clean.iloc[0:0]
    return pd.DataFrame(rows).sort_values("qed")


def simple_hypervolume_qed_sa(front: pd.DataFrame, reference_qed: float = 0.0, reference_sa: float = 10.0) -> float:
    """Approximate 2D hypervolume for maximizing QED and minimizing SA."""
    if front.empty:
        return 0.0
    f = front[["qed", "sa"]].dropna().sort_values("qed").copy()
    hv = 0.0
    prev_qed = reference_qed
    for _, row in f.iterrows():
        qed = max(float(row["qed"]), reference_qed)
        sa_gain = max(reference_sa - float(row["sa"]), 0.0)
        width = max(qed - prev_qed, 0.0)
        hv += width * sa_gain
        prev_qed = max(prev_qed, qed)
    return float(hv)


def coverage_qed_sa(df: pd.DataFrame, qed_cutoff: float = 0.7, sa_cutoff: float = 4.0) -> float:
    clean = df[["qed", "sa"]].dropna()
    if clean.empty:
        return float("nan")
    return float(((clean["qed"] >= qed_cutoff) & (clean["sa"] <= sa_cutoff)).mean() * 100.0)


def build_figure2_data(seed_data: list[SeedData], experiment_root: Path, out_dir: Path) -> dict[str, Path]:
    fig2_dir = out_dir / "figure2"
    fig2_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [sd.properties.assign(seed=sd.seed) for sd in seed_data],
        ignore_index=True,
    )

    landscape_cols = [
        c
        for c in [
            "seed",
            "latent_code_hex",
            "canonical_smiles",
            "smiles",
            "qed",
            "sa",
            "logp",
            "ro5",
            "representative_energy",
            "energies",
            "official_final_score",
            "official_property_contribution",
        ]
        if c in combined.columns
    ]
    landscape = combined[landscape_cols].dropna(subset=["qed", "sa"]).copy()
    landscape_path = fig2_dir / "figure2a_qed_sa_landscape.csv"
    landscape.to_csv(landscape_path, index=False)

    front_rows = []
    metric_rows = []
    for label, df in [("all", landscape)] + [(f"seed_{sd.seed}", sd.properties) for sd in seed_data]:
        front = pareto_front_qed_sa(df)
        if not front.empty:
            front.insert(0, "group", label)
            front_rows.append(front)
        metric_rows.append(
            {
                "group": label,
                "n": int(df[["qed", "sa"]].dropna().shape[0]),
                "hypervolume_qed_sa": simple_hypervolume_qed_sa(front),
                "coverage_qed_ge_0p7_sa_le_4p0_percent": coverage_qed_sa(df),
                "qed_mean": float(df["qed"].mean()) if "qed" in df else float("nan"),
                "sa_mean": float(df["sa"].mean()) if "sa" in df else float("nan"),
            }
        )
    front_path = fig2_dir / "figure2a_pareto_front.csv"
    if front_rows:
        pd.concat(front_rows, ignore_index=True).to_csv(front_path, index=False)
    else:
        pd.DataFrame(columns=["group", "qed", "sa"]).to_csv(front_path, index=False)
    metrics_path = fig2_dir / "figure2a_landscape_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    comparison = build_table1_comparison(experiment_root)
    comparison_path = fig2_dir / "figure2b_table1_paper_vs_repro.csv"
    comparison.to_csv(comparison_path, index=False)

    examples = select_molecule_examples(combined)
    examples_path = fig2_dir / "figure2c_representative_molecules.csv"
    examples.to_csv(examples_path, index=False)

    return {
        "figure2a_landscape": landscape_path,
        "figure2a_pareto_front": front_path,
        "figure2a_metrics": metrics_path,
        "figure2b_table1_comparison": comparison_path,
        "figure2c_examples": examples_path,
    }


def build_table1_comparison(experiment_root: Path) -> pd.DataFrame:
    aggregate = experiment_root / "aggregate" / "table1_paper_mean_std.csv"
    if aggregate.exists():
        repro = pd.read_csv(aggregate)
        repro["property"] = repro["property"].map(normalize_property)
        repro["subset"] = repro["subset"].map(normalize_subset)
    else:
        all_files = sorted(experiment_root.glob("seed_*/evaluation_paper_protocol/observed_table1.csv"))
        if not all_files:
            raise FileNotFoundError("Could not locate Table 1 aggregate or per-seed observed_table1.csv")
        frames = []
        for path in all_files:
            frame = pd.read_csv(path)
            frame["seed"] = path.parents[1].name.replace("seed_", "")
            frames.append(frame)
        all_df = pd.concat(frames, ignore_index=True)
        all_df["property"] = all_df["property"].map(normalize_property)
        all_df["subset"] = all_df["subset"].map(normalize_subset)
        repro = (
            all_df.groupby(["property", "subset"], as_index=False)["observed"]
            .agg(mean="mean", std="std", count="count")
            .reset_index(drop=True)
        )

    rows = []
    for prop, subset_values in PAPER_JANUS_FM.items():
        for subset, paper_value in subset_values.items():
            match = repro[(repro["property"] == prop) & (repro["subset"] == subset)]
            if match.empty:
                repro_mean = np.nan
                repro_std = np.nan
                count = 0
            else:
                record = match.iloc[0]
                repro_mean = float(record.get("mean", record.get("observed", np.nan)))
                repro_std = float(record.get("std", np.nan))
                count = int(record.get("count", 1))
            rows.append(
                {
                    "property": prop,
                    "subset": subset,
                    "paper_janus_fm": paper_value,
                    "repro_mean": repro_mean,
                    "repro_std": repro_std,
                    "seed_count": count,
                    "delta_repro_minus_paper": repro_mean - paper_value if np.isfinite(repro_mean) else np.nan,
                    "abs_delta": abs(repro_mean - paper_value) if np.isfinite(repro_mean) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def select_molecule_examples(df: pd.DataFrame, n: int = 24) -> pd.DataFrame:
    work = df.copy()
    if "canonical_smiles" not in work and "smiles" in work:
        work["canonical_smiles"] = work["smiles"]
    rank_cols = [
        c
        for c in ["official_final_score", "official_property_contribution", "qed", "score_qed"]
        if c in work.columns
    ]
    if rank_cols:
        work["_rank_score"] = work[rank_cols[0]]
        work = work.sort_values("_rank_score", ascending=False)
    elif "representative_energy" in work:
        work = work.sort_values("representative_energy", ascending=True)
    keep = [
        c
        for c in [
            "seed",
            "canonical_smiles",
            "smiles",
            "qed",
            "sa",
            "logp",
            "tpsa",
            "mw",
            "ro5",
            "official_final_score",
            "representative_energy",
        ]
        if c in work.columns
    ]
    return work.drop_duplicates("canonical_smiles").head(n)[keep]


def available_targets(df: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    out = []
    for target in requested:
        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            out.append(target)
    return out


def build_latent_model_data(
    seed_data: list[SeedData],
    out_dir: Path,
    dim: int,
    targets: list[str],
    top_bits: int,
    max_pair_bits: int,
    random_state: int,
) -> dict[str, Path]:
    fig3_dir = out_dir / "figure3"
    fig3_dir.mkdir(parents=True, exist_ok=True)
    bit_cols = [f"bit_{i}" for i in range(dim)]
    combined = pd.concat(
        [sd.properties.assign(seed=sd.seed) for sd in seed_data],
        ignore_index=True,
    )
    present_bits = [c for c in bit_cols if c in combined.columns]
    if len(present_bits) < dim:
        warnings.warn(
            f"Only {len(present_bits)} latent bit columns found. Figure 3/4 data will be limited."
        )
    target_cols = available_targets(combined, targets)
    if not target_cols:
        raise ValueError(f"No requested targets found in generated properties: {targets}")

    latent_export_cols = [
        c
        for c in ["seed", "latent_code_hex", "canonical_smiles", "smiles", *target_cols, *present_bits]
        if c in combined.columns
    ]
    latent_matrix_path = fig3_dir / "figure3_latent_property_matrix.csv"
    combined[latent_export_cols].dropna(subset=present_bits[:1] + target_cols[:1]).to_csv(
        latent_matrix_path, index=False
    )

    contributions = []
    pair_rows = []
    model_rows = []
    for target in target_cols:
        usable = combined[present_bits + [target]].dropna().copy()
        if usable.empty:
            continue
        y = usable[target].astype(float)
        global_mean = float(y.mean())
        for i, bit_col in enumerate(present_bits):
            state0 = usable.loc[usable[bit_col] == 0, target]
            state1 = usable.loc[usable[bit_col] == 1, target]
            if len(state0) == 0 or len(state1) == 0:
                continue
            mean0 = float(state0.mean())
            mean1 = float(state1.mean())
            effect = mean1 - mean0
            contributions.append(
                {
                    "target": target,
                    "bit_index": int(bit_col.split("_")[1]),
                    "bit": bit_col,
                    "count0": int(len(state0)),
                    "count1": int(len(state1)),
                    "mean_if_0": mean0,
                    "mean_if_1": mean1,
                    "contribution_if_0": mean0 - global_mean,
                    "contribution_if_1": mean1 - global_mean,
                    "effect_1_minus_0": effect,
                    "abs_effect": abs(effect),
                    "global_mean": global_mean,
                }
            )

        model_rows.extend(train_optional_regressor(usable[present_bits], y, target, random_state))

        contrib_df = pd.DataFrame([r for r in contributions if r["target"] == target])
        if contrib_df.empty:
            continue
        candidate_bits = (
            contrib_df.sort_values("abs_effect", ascending=False)
            .head(max_pair_bits)["bit"]
            .tolist()
        )
        pair_rows.extend(compute_pair_interactions(usable, target, candidate_bits))

    contrib_path = fig3_dir / "figure3_bit_contributions.csv"
    pd.DataFrame(contributions).to_csv(contrib_path, index=False)
    pair_path = fig3_dir / "figure3_pair_interactions.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    model_path = fig3_dir / "figure3_model_diagnostics.csv"
    pd.DataFrame(model_rows).to_csv(model_path, index=False)

    top_path = fig3_dir / "figure3_top_bits_by_target.csv"
    contrib_all = pd.DataFrame(contributions)
    if not contrib_all.empty:
        top = (
            contrib_all.sort_values(["target", "abs_effect"], ascending=[True, False])
            .groupby("target", as_index=False)
            .head(top_bits)
        )
    else:
        top = pd.DataFrame()
    top.to_csv(top_path, index=False)

    return {
        "figure3_latent_property_matrix": latent_matrix_path,
        "figure3_bit_contributions": contrib_path,
        "figure3_pair_interactions": pair_path,
        "figure3_model_diagnostics": model_path,
        "figure3_top_bits": top_path,
    }


def train_optional_regressor(X: pd.DataFrame, y: pd.Series, target: str, random_state: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(X) < 200:
        return rows
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return rows

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        y.astype(float),
        test_size=0.2,
        random_state=random_state,
    )

    model_name = None
    model = None
    try:
        import lightgbm as lgb

        model_name = "LightGBM"
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            model_name = "HistGradientBoostingRegressor"
            model = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                random_state=random_state,
            )
        except Exception:
            return rows

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, pred))
    rows.append(
        {
            "target": target,
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(math.sqrt(mse)),
        }
    )

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        total = float(importances.sum()) or 1.0
        for bit, value in zip(X.columns, importances):
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "metric": "feature_importance",
                    "bit": bit,
                    "bit_index": int(bit.split("_")[1]),
                    "importance": float(value),
                    "importance_fraction": float(value / total),
                }
            )
    return rows


def compute_pair_interactions(df: pd.DataFrame, target: str, bit_cols: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, bit_i in enumerate(bit_cols):
        for bit_j in bit_cols[idx + 1 :]:
            means: dict[str, float] = {}
            counts: dict[str, int] = {}
            for vi in (0, 1):
                for vj in (0, 1):
                    mask = (df[bit_i] == vi) & (df[bit_j] == vj)
                    values = df.loc[mask, target]
                    key = f"{vi}{vj}"
                    counts[key] = int(values.shape[0])
                    means[key] = float(values.mean()) if len(values) else np.nan
            if min(counts.values()) < 10:
                continue
            synergy = means["11"] - means["10"] - means["01"] + means["00"]
            best_state = max(means.items(), key=lambda kv: -np.inf if pd.isna(kv[1]) else kv[1])[0]
            rows.append(
                {
                    "target": target,
                    "bit_i": bit_i,
                    "bit_j": bit_j,
                    "bit_i_index": int(bit_i.split("_")[1]),
                    "bit_j_index": int(bit_j.split("_")[1]),
                    "mean_00": means["00"],
                    "mean_01": means["01"],
                    "mean_10": means["10"],
                    "mean_11": means["11"],
                    "count_00": counts["00"],
                    "count_01": counts["01"],
                    "count_10": counts["10"],
                    "count_11": counts["11"],
                    "synergy_11_minus_additive": float(synergy),
                    "abs_synergy": float(abs(synergy)),
                    "best_state": best_state,
                    "best_state_mean": means[best_state],
                }
            )
    return sorted(rows, key=lambda r: r["abs_synergy"], reverse=True)


def build_figure4_data(
    seed_data: list[SeedData],
    out_dir: Path,
    dim: int,
    target_for_bits: str,
    top_bits: int,
    contrib_path: Path,
) -> dict[str, Path]:
    fig4_dir = out_dir / "figure4"
    fig4_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [sd.properties.assign(seed=sd.seed) for sd in seed_data],
        ignore_index=True,
    )
    bit_cols = [f"bit_{i}" for i in range(dim) if f"bit_{i}" in combined.columns]

    contrib = pd.read_csv(contrib_path) if contrib_path.exists() else pd.DataFrame()
    if not contrib.empty and target_for_bits in set(contrib["target"]):
        selected_bits = (
            contrib[contrib["target"] == target_for_bits]
            .sort_values("abs_effect", ascending=False)
            .head(top_bits)["bit"]
            .tolist()
        )
    elif bit_cols:
        selected_bits = bit_cols[:top_bits]
    else:
        selected_bits = []

    property_cols = [
        c
        for c in [
            "qed",
            "sa",
            "logp",
            "mw",
            "tpsa",
            "nrb",
            "hbd",
            "hba",
            "ro5",
            "official_property_contribution",
            "official_final_score",
        ]
        if c in combined.columns and pd.api.types.is_numeric_dtype(combined[c])
    ]
    property_rows = []
    for bit in selected_bits:
        for prop in property_cols:
            state0 = combined.loc[combined[bit] == 0, prop].dropna()
            state1 = combined.loc[combined[bit] == 1, prop].dropna()
            if len(state0) == 0 or len(state1) == 0:
                continue
            mean0 = float(state0.mean())
            mean1 = float(state1.mean())
            property_rows.append(
                {
                    "bit": bit,
                    "bit_index": int(bit.split("_")[1]),
                    "property": prop,
                    "count0": int(len(state0)),
                    "count1": int(len(state1)),
                    "mean_if_0": mean0,
                    "mean_if_1": mean1,
                    "shift_1_minus_0": mean1 - mean0,
                    "relative_shift_percent": ((mean1 - mean0) / abs(mean0) * 100.0)
                    if abs(mean0) > 1e-12
                    else np.nan,
                }
            )
    property_shift_path = fig4_dir / "figure4_property_shift_by_bit.csv"
    pd.DataFrame(property_rows).to_csv(property_shift_path, index=False)

    substructure_path = fig4_dir / "figure4_substructure_shift_by_bit.csv"
    substructure_df = compute_substructure_shifts(combined, selected_bits)
    substructure_df.to_csv(substructure_path, index=False)

    distribution_cols = [
        c
        for c in ["seed", "canonical_smiles", "smiles", *property_cols]
        if c in combined.columns
    ]
    distribution_path = fig4_dir / "figure4_property_distributions.csv"
    combined[distribution_cols].to_csv(distribution_path, index=False)

    return {
        "figure4_property_shift": property_shift_path,
        "figure4_substructure_shift": substructure_path,
        "figure4_property_distributions": distribution_path,
    }


def compute_substructure_shifts(df: pd.DataFrame, selected_bits: list[str]) -> pd.DataFrame:
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    if smiles_col not in df.columns or not selected_bits:
        return pd.DataFrame()
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        warnings.warn("RDKit is not available; skipping substructure shifts.")
        return pd.DataFrame()

    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None for s in df[smiles_col]]
    pattern_mols = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS_PATTERNS.items()}
    flags = pd.DataFrame(index=df.index)
    for name, pattern in pattern_mols.items():
        if pattern is None:
            continue
        flags[name] = [bool(mol is not None and mol.HasSubstructMatch(pattern)) for mol in mols]

    rows = []
    for bit in selected_bits:
        if bit not in df.columns:
            continue
        mask0 = df[bit] == 0
        mask1 = df[bit] == 1
        for name in flags.columns:
            freq0 = float(flags.loc[mask0, name].mean() * 100.0) if mask0.any() else np.nan
            freq1 = float(flags.loc[mask1, name].mean() * 100.0) if mask1.any() else np.nan
            rows.append(
                {
                    "bit": bit,
                    "bit_index": int(bit.split("_")[1]),
                    "substructure": name,
                    "frequency_if_0_percent": freq0,
                    "frequency_if_1_percent": freq1,
                    "shift_percentage_points": freq1 - freq0,
                    "count0": int(mask0.sum()),
                    "count1": int(mask1.sum()),
                }
            )
    return pd.DataFrame(rows)


def make_plots(out_dir: Path, max_points: int = 60000, random_state: int = 42) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.warn("matplotlib is not available; skipping PNG plot generation.")
        return []

    paths: list[Path] = []
    fig2_dir = out_dir / "figure2"
    landscape_path = fig2_dir / "figure2a_qed_sa_landscape.csv"
    front_path = fig2_dir / "figure2a_pareto_front.csv"
    if landscape_path.exists():
        landscape = pd.read_csv(landscape_path)
        if len(landscape) > max_points:
            landscape = landscape.sample(max_points, random_state=random_state)
        fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=180)
        hb = ax.hexbin(
            landscape["qed"],
            landscape["sa"],
            gridsize=55,
            cmap="Greens",
            mincnt=1,
            linewidths=0.0,
        )
        ax.scatter(landscape["qed"], landscape["sa"], s=2, c="black", alpha=0.08)
        if front_path.exists():
            front = pd.read_csv(front_path)
            front = front[front["group"] == "all"] if "group" in front.columns else front
            if not front.empty:
                ax.plot(front["qed"], front["sa"], color="#c0392b", lw=1.8, marker="o", ms=2.5)
        ax.set_xlabel("QED")
        ax.set_ylabel("SA score (lower is better)")
        ax.set_title("Figure 2a-style QED-SA landscape")
        ax.invert_yaxis()
        fig.colorbar(hb, ax=ax, label="molecule density")
        fig.tight_layout()
        path = fig2_dir / "figure2a_qed_sa_landscape.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    comparison_path = fig2_dir / "figure2b_table1_paper_vs_repro.csv"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180)
        axes = axes.ravel()
        for ax, prop in zip(axes, ["QED", "SA", "LogP", "Ro5"]):
            sub = comp[comp["property"] == prop].copy()
            x = np.arange(len(sub))
            width = 0.38
            ax.bar(x - width / 2, sub["paper_janus_fm"], width, label="Paper Janus-FM", color="#4c72b0")
            ax.bar(
                x + width / 2,
                sub["repro_mean"],
                width,
                yerr=sub["repro_std"].fillna(0.0),
                label="Reproduction",
                color="#dd8452",
                capsize=2,
            )
            ax.set_title(prop)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["subset"])
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Figure 2b-style Table 1 comparison", y=1.01)
        fig.tight_layout()
        path = fig2_dir / "figure2b_table1_paper_vs_repro.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig3_dir = out_dir / "figure3"
    contrib_path = fig3_dir / "figure3_bit_contributions.csv"
    if contrib_path.exists():
        contrib = pd.read_csv(contrib_path)
        if not contrib.empty:
            target = "official_property_contribution"
            if target not in set(contrib["target"]):
                target = str(contrib["target"].iloc[0])
            sub = contrib[contrib["target"] == target].copy()
            fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=180)
            sc = ax.scatter(
                sub["contribution_if_0"],
                sub["contribution_if_1"],
                c=sub["abs_effect"],
                cmap="viridis",
                s=28,
                alpha=0.85,
            )
            ax.axhline(0, color="0.4", lw=0.8)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.set_xlabel("Contribution when bit = 0")
            ax.set_ylabel("Contribution when bit = 1")
            ax.set_title(f"Figure 3b-style bit contributions ({target})")
            fig.colorbar(sc, ax=ax, label="|effect|")
            fig.tight_layout()
            path = fig3_dir / "figure3b_bit_contribution_scatter.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    fig4_dir = out_dir / "figure4"
    prop_shift_path = fig4_dir / "figure4_property_shift_by_bit.csv"
    if prop_shift_path.exists():
        shifts = pd.read_csv(prop_shift_path)
        if not shifts.empty:
            props = ["qed", "sa", "logp", "mw", "tpsa", "nrb", "hbd", "hba"]
            piv = (
                shifts[shifts["property"].isin(props)]
                .pivot_table(index="bit", columns="property", values="shift_1_minus_0", aggfunc="mean")
                .fillna(0.0)
            )
            if not piv.empty:
                fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.36 * len(piv))), dpi=180)
                im = ax.imshow(piv.values, aspect="auto", cmap="coolwarm")
                ax.set_xticks(np.arange(len(piv.columns)))
                ax.set_xticklabels(piv.columns, rotation=45, ha="right")
                ax.set_yticks(np.arange(len(piv.index)))
                ax.set_yticklabels(piv.index)
                ax.set_title("Figure 4-style property shifts by latent bit")
                fig.colorbar(im, ax=ax, label="mean(bit=1) - mean(bit=0)")
                fig.tight_layout()
                path = fig4_dir / "figure4_property_shift_heatmap.png"
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)

    return paths


def write_manifest(out_dir: Path, args: argparse.Namespace, outputs: dict[str, Path], plot_paths: list[Path]) -> None:
    manifest = {
        "script": Path(__file__).name,
        "experiment_root": str(args.experiment_root),
        "protocol_dir": args.protocol_dir,
        "latent_dim": args.latent_dim,
        "targets": args.targets,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "plots": [str(path) for path in plot_paths],
        "notes": [
            "Figure 2 Table 1 comparison uses paper Janus-FM values from the main manuscript.",
            "Figure 3 contributions are empirical bit-state effects; LightGBM diagnostics are added when available.",
            "Figure 4 shifts compare generated molecules with bit=1 versus bit=0, not original-vs-optimized pairs.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/paper_artifact"),
        help="Experiment A root directory.",
    )
    parser.add_argument(
        "--protocol-dir",
        default=DEFAULT_PROTOCOL_DIR,
        help="Per-seed evaluation directory to read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to EXPERIMENT_ROOT/figure_data.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional seed list, e.g. --seeds 42 123 2026.",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--top-bits", type=int, default=16)
    parser.add_argument("--max-pair-bits", type=int, default=20)
    parser.add_argument(
        "--figure4-target",
        default="official_property_contribution",
        help="Target used to choose important bits for Figure 4 shifts.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/JSON data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root
    out_dir = args.output_dir or (experiment_root / "figure_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_data = load_seed_data(
        experiment_root=experiment_root,
        protocol_dir=args.protocol_dir,
        seeds=args.seeds,
        dim=args.latent_dim,
    )
    outputs: dict[str, Path] = {}
    outputs.update(build_figure2_data(seed_data, experiment_root, out_dir))
    outputs.update(
        build_latent_model_data(
            seed_data=seed_data,
            out_dir=out_dir,
            dim=args.latent_dim,
            targets=args.targets,
            top_bits=args.top_bits,
            max_pair_bits=args.max_pair_bits,
            random_state=args.random_state,
        )
    )
    outputs.update(
        build_figure4_data(
            seed_data=seed_data,
            out_dir=out_dir,
            dim=args.latent_dim,
            target_for_bits=args.figure4_target,
            top_bits=args.top_bits,
            contrib_path=outputs["figure3_bit_contributions"],
        )
    )
    plot_paths = [] if args.no_plots else make_plots(out_dir, random_state=args.random_state)
    write_manifest(out_dir, args, outputs, plot_paths)

    print(f"Wrote figure data to: {out_dir}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    for path in plot_paths:
        print(f"  plot: {path}")


if __name__ == "__main__":
    main()

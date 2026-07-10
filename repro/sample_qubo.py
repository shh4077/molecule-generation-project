"""Pure NumPy QUBO sampler for Janus 128-bit latent codes.

The upstream repo includes GA/TPE samplers. This local sampler has no DEAP or
Optuna dependency and is good for smoke tests and reproducible comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def energy(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", X, Q, X)


def anneal(
    Q: np.ndarray,
    n_chains: int,
    sweeps: int,
    seed: int,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_dim = Q.shape[0]
    X = rng.integers(0, 2, size=(n_chains, n_dim), dtype=np.int8)
    E = energy(Q, X.astype(np.float64))
    best_X = X.copy()
    best_E = E.copy()

    off = Q + Q.T
    np.fill_diagonal(off, 0.0)
    diag = np.diag(Q)

    for sweep in tqdm(range(sweeps), desc="QUBO annealing"):
        frac = sweep / max(1, sweeps - 1)
        temp = t_start * ((t_end / t_start) ** frac)
        bit_order = rng.permutation(n_dim)
        for bit in bit_order:
            X_float = X.astype(np.float64)
            contribution_on = diag[bit] + X_float @ off[:, bit]
            delta = np.where(X[:, bit] == 0, contribution_on, -contribution_on)
            uphill = np.maximum(delta, 0.0)
            accept_prob = np.exp(-uphill / max(temp, 1e-8))
            accept = (delta <= 0.0) | (rng.random(n_chains) < accept_prob)
            X[accept, bit] = 1 - X[accept, bit]
            E[accept] += delta[accept]
            improved = E < best_E
            best_E[improved] = E[improved]
            best_X[improved] = X[improved]

    return best_X, best_E


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample low-energy binary latent codes from a QUBO CSV.")
    parser.add_argument("--qubo-csv", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-chains", type=int, default=20_000)
    parser.add_argument("--sweeps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-start", type=float, default=2.0)
    parser.add_argument("--t-end", type=float, default=0.01)
    args = parser.parse_args()

    Q = np.loadtxt(args.qubo_csv, delimiter=",").astype(np.float64)
    samples, energies = anneal(Q, args.n_chains, args.sweeps, args.seed, args.t_start, args.t_end)

    unique_samples, unique_idx = np.unique(samples, axis=0, return_index=True)
    unique_energies = energies[unique_idx]
    order = np.argsort(unique_energies)
    order = order[: args.n_samples]
    selected = unique_samples[order].astype(np.int8)
    selected_energies = unique_energies[order].astype(np.float32)

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output_h5, "w") as h5:
        h5.create_dataset("latent_codes", data=selected, compression="gzip")
        h5.create_dataset("energies", data=selected_energies, compression="gzip")
        h5.create_dataset(
            "labels",
            data=[f"energy={x:.6f}" for x in selected_energies],
            dtype=str_dtype,
            compression="gzip",
        )
        h5.attrs["qubo_csv"] = str(args.qubo_csv)
        h5.attrs["n_unique"] = len(unique_samples)

    print(f"Wrote {args.output_h5}")
    print(f"Unique candidates: {len(unique_samples):,}; selected: {len(selected):,}")
    print(f"Best energy: {float(selected_energies[0]):.6f}")


if __name__ == "__main__":
    main()

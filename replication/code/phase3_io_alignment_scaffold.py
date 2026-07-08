# -*- coding: utf-8 -*-
"""Phase 3 scaffold: alignment between news lead-lag networks and the
input-output production structure.

STATUS: code ready; requires an ICIO coefficient matrix to run.

Data requirement (choose one, place in this folder):
  A) OECD ICIO bulk table (https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html),
     one reference year. Iran is not a separately listed economy in recent
     editions; options: (i) use the Rest-of-World block, (ii) use world-average
     domestic coefficients, (iii) use a comparable single economy as proxy.
     Aggregate to the 50-sector V1 scheme using icio_sectors_with_isic_rev4_descriptions.csv.
  B) Any 50x50 technical-coefficient matrix A saved as `icio_A_50x50.csv`
     (rows=input sector V1 1..50, cols=output sector V1 1..50).

Analyses implemented below (run once A is available):
  1. Edge concentration: are FDR-significant news edges (a->b) more frequent
     among IO-adjacent pairs (a_ij above quantile q) than non-adjacent pairs?
     Fisher exact / logistic regression with |r| as covariate.
  2. QAP correlation: permutation test correlating the news adjacency matrix
     with A and with the Leontief inverse L=(I-A)^-1 (row/col permutations).
  3. Propagation loop: news shock vector eps_t (daily share vector) propagated
     through L; correlate L@eps_t with observed news exposure at t+k, k=1..7.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
NET_DIR = BASE / "figures" / "causal network_v4_share_fdr"
A_FILE = BASE / "icio_A_50x50.csv"   # <- provide this


def leontief(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.eye(A.shape[0]) - A)


def qap_corr(M1: np.ndarray, M2: np.ndarray, n_perm: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Off-diagonal Pearson correlation of two square matrices + QAP p-value."""
    n = M1.shape[0]
    mask = ~np.eye(n, dtype=bool)
    obs = np.corrcoef(M1[mask], M2[mask])[0, 1]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Mp = M2[np.ix_(perm, perm)]
        if abs(np.corrcoef(M1[mask], Mp[mask])[0, 1]) >= abs(obs):
            cnt += 1
    return obs, cnt / n_perm


def main() -> None:
    if not A_FILE.exists():
        raise SystemExit(
            f"Missing {A_FILE.name}. See module docstring for how to obtain "
            "and aggregate an ICIO coefficient matrix to the V1 scheme."
        )
    A = pd.read_csv(A_FILE, index_col=0).values.astype(float)
    L = leontief(A)

    for lang in ["arabic", "chinese", "english", "persian"]:
        edges = pd.read_csv(NET_DIR / f"ty_edges_allpairs_{lang}.csv")
        sig = edges[edges["fdr_significant"]]
        N = np.zeros((50, 50))
        for r in sig.itertuples(index=False):
            N[int(r.source) - 1, int(r.target) - 1] = 1.0

        # 1) edge concentration among IO-adjacent pairs (top-quartile a_ij)
        q75 = np.quantile(A[A > 0], 0.75) if (A > 0).any() else 0
        adj = (A >= q75).astype(float)
        mask = ~np.eye(50, dtype=bool)
        both = int(((N == 1) & (adj == 1))[mask].sum())
        news_only = int(((N == 1) & (adj == 0))[mask].sum())
        io_only = int(((N == 0) & (adj == 1))[mask].sum())
        neither = int(((N == 0) & (adj == 0))[mask].sum())
        from scipy.stats import fisher_exact
        orr, p = fisher_exact([[both, news_only], [io_only, neither]])

        # 2) QAP against A and L
        r_A, p_A = qap_corr(N, A)
        r_L, p_L = qap_corr(N, L)
        print(f"{lang}: edges={int(N.sum())}  Fisher OR={orr:.2f} (p={p:.3f})  "
              f"QAP r(A)={r_A:.3f} (p={p_A:.3f})  QAP r(L)={r_L:.3f} (p={p_L:.3f})")

        # 3) propagation loop
        sh = pd.read_csv(NET_DIR / f"daily_share_exuniform_{lang}.csv",
                         index_col=0, parse_dates=True)
        X = sh.values  # T x 50
        pred = X @ L.T  # Leontief-propagated exposure
        for k in [1, 3, 7]:
            r = np.corrcoef(pred[:-k].ravel(), X[k:].ravel())[0, 1]
            print(f"   propagation corr(L@eps_t, eps_t+{k}) = {r:.3f}")


if __name__ == "__main__":
    main()

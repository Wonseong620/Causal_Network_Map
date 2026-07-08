# -*- coding: utf-8 -*-
"""Phase 3 (IO alignment) re-run on the LLM-based FDR network
(figures/causal_network_v5_llm_fdr/), reusing the world-average coefficient
matrix built in build_world_A.py (icio_A_50x50.csv).

Same analyses as phase3_io_alignment_scaffold.py: edge concentration among
IO-adjacent pairs (Fisher exact), QAP correlation against A and the Leontief
inverse L, and a Leontief-propagation check against the LLM daily share
series.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE = Path(__file__).resolve().parent
NET_DIR = BASE / "figures" / "causal_network_v5_llm_fdr"
A_FILE = BASE / "icio_A_50x50.csv"


def leontief(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.eye(A.shape[0]) - A)


def qap_corr(M1: np.ndarray, M2: np.ndarray, n_perm: int = 5000, seed: int = 0):
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
    A = pd.read_csv(A_FILE, index_col=0).values.astype(float)
    L = leontief(A)
    rows = []

    for lang in ["arabic", "chinese", "english", "persian"]:
        edges = pd.read_csv(NET_DIR / f"ty_edges_allpairs_llm_{lang}.csv")
        sig = edges[edges["fdr_significant"]]
        N = np.zeros((50, 50))
        for r in sig.itertuples(index=False):
            N[int(r.source) - 1, int(r.target) - 1] = 1.0

        q75 = np.quantile(A[A > 0], 0.75) if (A > 0).any() else 0
        adj = (A >= q75).astype(float)
        mask = ~np.eye(50, dtype=bool)
        both = int(((N == 1) & (adj == 1))[mask].sum())
        news_only = int(((N == 1) & (adj == 0))[mask].sum())
        io_only = int(((N == 0) & (adj == 1))[mask].sum())
        neither = int(((N == 0) & (adj == 0))[mask].sum())
        orr, p = fisher_exact([[both, news_only], [io_only, neither]])

        r_A, p_A = qap_corr(N, A)
        r_L, p_L = qap_corr(N, L)

        sh = pd.read_csv(NET_DIR / f"daily_share_llm_{lang}.csv", index_col=0, parse_dates=True)
        X = sh.values
        pred = X @ L.T
        prop = {}
        for k in [1, 3, 7]:
            prop[k] = float(np.corrcoef(pred[:-k].ravel(), X[k:].ravel())[0, 1])

        print(f"{lang}: edges={int(N.sum())}  Fisher OR={orr:.2f} (p={p:.3f})  "
              f"QAP r(A)={r_A:.3f} (p={p_A:.3f})  QAP r(L)={r_L:.3f} (p={p_L:.3f})  "
              f"propagation k=1/3/7: {prop[1]:.3f}/{prop[3]:.3f}/{prop[7]:.3f}")
        rows.append({"language": lang, "edges": int(N.sum()), "fisher_or": round(orr, 3),
                    "fisher_p": round(p, 4), "qap_r_A": round(r_A, 3), "qap_p_A": round(p_A, 4),
                    "qap_r_L": round(r_L, 3), "qap_p_L": round(p_L, 4),
                    "prop_k1": round(prop[1], 3), "prop_k3": round(prop[3], 3), "prop_k7": round(prop[7], 3)})

    pd.DataFrame(rows).to_csv(NET_DIR / "io_alignment_llm.csv", index=False)
    print("saved", NET_DIR / "io_alignment_llm.csv")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Phase 3 (IO alignment) on the common-factor-adjusted LLM network
(figures/causal_network_v5c_llm_cfadj/)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE = Path(__file__).resolve().parent
NET_DIR = BASE / "figures" / "causal_network_v5c_llm_cfadj"
A_FILE = BASE / "icio_A_50x50.csv"


def leontief(A):
    return np.linalg.inv(np.eye(A.shape[0]) - A)


def qap_corr(M1, M2, n_perm=5000, seed=0):
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


def main():
    A = pd.read_csv(A_FILE, index_col=0).values.astype(float)
    L = leontief(A)
    rows = []
    for lang in ["arabic", "chinese", "english", "persian"]:
        edges = pd.read_csv(NET_DIR / f"ty_edges_allpairs_cfadj_{lang}.csv")
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

        print(f"{lang}: edges={int(N.sum())}  Fisher OR={orr:.2f} (p={p:.3f})  "
              f"QAP r(A)={r_A:.3f} (p={p_A:.3f})  QAP r(L)={r_L:.3f} (p={p_L:.3f})")
        rows.append({"language": lang, "edges": int(N.sum()), "fisher_or": round(orr, 3),
                    "fisher_p": round(p, 4), "qap_r_A": round(r_A, 3), "qap_p_A": round(p_A, 4),
                    "qap_r_L": round(r_L, 3), "qap_p_L": round(p_L, 4)})
    pd.DataFrame(rows).to_csv(NET_DIR / "io_alignment_cfadj.csv", index=False)
    print("saved", NET_DIR / "io_alignment_cfadj.csv")


if __name__ == "__main__":
    main()

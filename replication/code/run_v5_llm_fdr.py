# -*- coding: utf-8 -*-
"""v5 pipeline: full-corpus LLM sector probabilities (figures/llm_full/)
through the corrected indicator + network methodology of the v3 plan.

Stage 1: join LLM probabilities to article metadata (published_at,
  negativity, war_related) by orig_index; aggregate V/H indicators;
  build daily within-day-share series (no uniform exclusion needed:
  the LLM run produced zero low-evidence rows).
Stage 2: pairwise Toda-Yamamoto over all 2,450 ordered pairs per
  language on the share series; Benjamini-Hochberg FDR at 5%.
Stage 3: IO-structure alignment (Fisher / QAP vs world-average A and
  Leontief inverse; propagation correlations) using icio_A_50x50.csv.

Outputs -> figures/causal network_v5_llm_fdr/
"""
import io, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
TY_DIR = BASE / "figures" / "causal network_html_v4" / "html_generation_csv_and_code"
sys.path.insert(0, str(TY_DIR))
from create_ty_v1_networks import (  # noqa: E402
    integration_order, choose_var_lag, ty_wald_pvalue, best_positive_lead_corr,
)

LLM_DIR = BASE / "figures" / "llm_full"
OUT = BASE / "figures" / "causal network_v5_llm_fdr"
OUT.mkdir(parents=True, exist_ok=True)

PROB = [f"IO_V1_{i}" for i in range(1, 51)]
LANGS = {
    "arabic": "iran_ar_v3_with_io_matrix.csv",
    "chinese": "iran_ch_v3_with_io_matrix.csv",
    "english": "iran_en_v3_with_io_matrix.csv",
    "persian": "iran_pe_v3_with_io_matrix.csv",
}
MAX_LAG = 7
FDR_Q = 0.05


def bh_fdr(pvals: np.ndarray, q: float) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, n + 1) / n)
    below = pvals[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if below.any():
        keep[order[: np.where(below)[0].max() + 1]] = True
    return keep


summary_rows = []
indicator_frames = []

for lang, meta_file in LANGS.items():
    t0 = time.time()
    print(f"=== {lang} ===")
    probs = pd.read_csv(LLM_DIR / f"llm_probs_full_{lang}.csv")
    meta = pd.read_csv(BASE / meta_file,
                       usecols=["published_at", "negativity", "war_related"],
                       low_memory=False)
    df = probs.join(meta, on="orig_index", how="left", validate="1:1")
    dates = pd.to_datetime(df["published_at"], errors="coerce", dayfirst=True).dt.floor("D")
    ok = dates.notna()
    df, dates = df[ok], dates[ok]
    P = df[PROB].to_numpy(dtype=float)
    neg = pd.to_numeric(df["negativity"], errors="coerce").fillna(0.0)
    war = pd.to_numeric(df["war_related"], errors="coerce").fillna(0.0)
    print(f"articles joined={len(df)} (low_evidence={int(df.low_evidence.sum())})")

    # ---- Stage 1a: aggregate indicators ----
    V = P.sum(axis=0)
    H = (P * (neg * war).to_numpy()[:, None]).sum(axis=0)
    ind = pd.DataFrame({
        "V1": range(1, 51), "V_total": V, "H_neg_hostility": H,
        "V_share": V / V.sum(), "H_share": H / H.sum(),
    })
    ind.insert(0, "language", lang)
    ind.to_csv(OUT / f"indicators_llm_{lang}.csv", index=False)
    indicator_frames.append(ind)

    # ---- Stage 1b: daily share series ----
    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = pd.DataFrame(P, index=dates.values, columns=range(1, 51)) \
        .groupby(level=0).sum().reindex(idx, fill_value=0.0)
    tot = daily.sum(axis=1)
    shares = daily.div(tot.replace(0, np.nan), axis=0).fillna(0.0)
    shares.index.name = "date"
    shares.to_csv(OUT / f"daily_share_llm_{lang}.csv")

    X = shares.values
    active = X.std(0) > 0
    Z = (X[:, active] - X[:, active].mean(0)) / X[:, active].std(0)
    C = np.corrcoef(Z.T)
    ev = np.linalg.eigvalsh(C)
    pc1 = ev.max() / ev.sum() * 100
    iu = np.triu_indices(C.shape[0], 1)
    mabsr = np.abs(C[iu]).mean()
    print(f"share series: PC1={pc1:.1f}%  mean|r|={mabsr:.3f}")

    # ---- Stage 2: TY + FDR ----
    nodes = [s for s in range(1, 51) if shares[s].nunique() > 1]
    orders = {s: integration_order(shares[s]) for s in nodes}
    rows = []
    for i, a in enumerate(nodes):
        for b in nodes:
            if a == b:
                continue
            pair = shares[[a, b]].copy()
            pair.columns = ["source", "target"]
            p_lag = choose_var_lag(pair, MAX_LAG)
            dmax = max(orders[a], orders[b])
            pv = ty_wald_pvalue(pair, "source", "target", p_lag, dmax)
            if not np.isfinite(pv):
                continue
            lead, r = best_positive_lead_corr(shares[a], shares[b])
            rows.append((a, b, p_lag, dmax, lead, pv, r, abs(r)))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(nodes)} sources ({time.time()-t0:.0f}s)")
    edges = pd.DataFrame(rows, columns=["source", "target", "p_lag", "dmax",
                                        "lead_days", "ty_pvalue", "pearson_r", "abs_r"])
    edges["fdr_significant"] = bh_fdr(edges["ty_pvalue"].values, FDR_Q)
    edges.insert(0, "language", lang)
    edges.to_csv(OUT / f"ty_edges_allpairs_{lang}.csv", index=False)

    n_fdr = int(edges["fdr_significant"].sum())
    fdr = edges[edges["fdr_significant"]]
    print(f"tested={len(edges)}  raw p<.05={int((edges.ty_pvalue<0.05).sum())}  "
          f"BH-FDR(5%)={n_fdr}  elapsed={time.time()-t0:.0f}s")
    summary_rows.append({
        "language": lang, "articles": len(df),
        "PC1_pct": round(pc1, 1), "mean_abs_r": round(mabsr, 3),
        "raw_p05": int((edges.ty_pvalue < 0.05).sum()), "fdr_5pct": n_fdr,
        "fdr_mean_abs_r": round(float(fdr.abs_r.mean()), 3) if n_fdr else np.nan,
        "fdr_neg_r": int((fdr.pearson_r < 0).sum()),
        "fdr_median_lead": float(fdr.lead_days.median()) if n_fdr else np.nan,
    })

pd.concat(indicator_frames).to_csv(OUT / "indicators_llm_all.csv", index=False)
summ = pd.DataFrame(summary_rows)
summ.to_csv(OUT / "summary_v5_llm_fdr.csv", index=False)
print("\n===== NETWORK SUMMARY =====")
print(summ.to_string(index=False))

# ---- Stage 3: IO alignment ----
A_FILE = BASE / "icio_A_50x50.csv"
if A_FILE.exists():
    from scipy.stats import fisher_exact
    A = pd.read_csv(A_FILE, index_col=0).values.astype(float)
    L = np.linalg.inv(np.eye(50) - A)
    mask = ~np.eye(50, dtype=bool)
    q75 = np.quantile(A[A > 0], 0.75)
    adj = (A >= q75).astype(float)
    rng = np.random.default_rng(0)

    def qap(N, M, n_perm=2000):
        obs = np.corrcoef(N[mask], M[mask])[0, 1]
        cnt = sum(
            abs(np.corrcoef(N[mask], M[np.ix_(p, p)][mask])[0, 1]) >= abs(obs)
            for p in (rng.permutation(50) for _ in range(n_perm)))
        return obs, cnt / n_perm

    print("\n===== IO ALIGNMENT (world-average A, ICIO 2018) =====")
    align_rows = []
    for lang in LANGS:
        e = pd.read_csv(OUT / f"ty_edges_allpairs_{lang}.csv")
        sig = e[e.fdr_significant]
        N = np.zeros((50, 50))
        for r in sig.itertuples(index=False):
            N[int(r.source) - 1, int(r.target) - 1] = 1.0
        both = int(((N == 1) & (adj == 1))[mask].sum())
        news_only = int(((N == 1) & (adj == 0))[mask].sum())
        io_only = int(((N == 0) & (adj == 1))[mask].sum())
        neither = int(((N == 0) & (adj == 0))[mask].sum())
        orr, p = fisher_exact([[both, news_only], [io_only, neither]])
        r_A, p_A = qap(N, A)
        r_L, p_L = qap(N, L)
        print(f"{lang}: edges={int(N.sum())}  Fisher OR={orr:.2f} (p={p:.3f})  "
              f"QAP r(A)={r_A:.3f} (p={p_A:.3f})  QAP r(L)={r_L:.3f} (p={p_L:.3f})")
        align_rows.append({"language": lang, "edges": int(N.sum()),
                           "fisher_or": round(orr, 2), "fisher_p": round(p, 4),
                           "qap_r_A": round(r_A, 3), "qap_p_A": round(p_A, 3),
                           "qap_r_L": round(r_L, 3), "qap_p_L": round(p_L, 3)})
    pd.DataFrame(align_rows).to_csv(OUT / "io_alignment_v5.csv", index=False)
else:
    print("icio_A_50x50.csv not found - skipping Stage 3")

print("\nDONE ->", OUT)

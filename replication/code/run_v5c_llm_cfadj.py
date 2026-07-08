# -*- coding: utf-8 -*-
"""Common-factor-adjusted re-estimation on LLM-classified probabilities.

Diagnosis (run_v5_llm_indicators_fdr.py + run_v5b/CLR checks) ruled out
V1=45 dominance, compositional closure, and daily sparsity as explanations
for the high PC1 (44-56%) in the LLM-based share series. The remaining
interpretation: a genuine latent "hostility intensity" factor jointly moves
several economically linked sectors on high-news-volume days, which is
plausible (LLM understands when one event implicates several sectors) but
contaminates pairwise lead-lag testing with common-driver correlation.

This script extracts the first principal component of each language's daily
share series as an explicit common-factor score, regresses it out of every
sector's series (OLS, keep residuals), verifies PC1 drops on the residual
matrix, and reruns the TY all-pairs test + BH-FDR on the residuals. Output
-> figures/causal_network_v5c_llm_cfadj/
"""
import sys, io, time
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

OUT = BASE / "figures" / "causal_network_v5c_llm_cfadj"
OUT.mkdir(parents=True, exist_ok=True)
LLM_DIR = BASE / "figures" / "llm_full"

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
    ranked = pvals[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    keep = np.zeros(n, dtype=bool)
    if below.any():
        kmax = np.where(below)[0].max()
        keep[order[: kmax + 1]] = True
    return keep


def pc1_meanr(M: np.ndarray):
    active = M.std(0) > 1e-12
    Z = (M[:, active] - M[:, active].mean(0)) / M[:, active].std(0)
    C = np.corrcoef(Z.T)
    ev = np.linalg.eigvalsh(C)
    pc1 = ev.max() / ev.sum() * 100
    iu = np.triu_indices(C.shape[0], 1)
    return pc1, np.abs(C[iu]).mean()


summary_rows = []

for lang, fname in LANGS.items():
    t0 = time.time()
    print(f"=== {lang} ===")
    meta = pd.read_csv(BASE / fname, usecols=["published_at"], low_memory=False)
    meta["published_at"] = pd.to_datetime(meta["published_at"], errors="coerce", dayfirst=True).dt.floor("D")
    llm = pd.read_csv(LLM_DIR / f"llm_probs_full_{lang}.csv").set_index("orig_index")
    df = meta.join(llm, how="inner")
    dates = df["published_at"]
    ok = dates.notna()
    df, dates = df[ok], dates[ok]
    P = df[PROB]

    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = P.groupby(dates.values).sum().reindex(idx, fill_value=0.0)
    daily.columns = list(range(1, 51))
    tot = daily.sum(axis=1)
    shares = daily.div(tot.replace(0, np.nan), axis=0).fillna(0.0)

    X = shares.values  # T x 50
    pc1_before, mr_before = pc1_meanr(X)

    # extract common-factor score: PC1 of standardized shares
    active = X.std(0) > 1e-12
    Xa = X[:, active]
    Za = (Xa - Xa.mean(0)) / Xa.std(0)
    C = np.corrcoef(Za.T)
    ev, evec = np.linalg.eigh(C)
    top = evec[:, np.argmax(ev)]
    factor = Za @ top  # T-vector, the common hostility-intensity score

    # regress out the factor from every (all 50) raw share series
    Fd = np.column_stack([np.ones_like(factor), factor])
    beta, *_ = np.linalg.lstsq(Fd, X, rcond=None)
    resid = X - Fd @ beta

    pc1_after, mr_after = pc1_meanr(resid)
    print(f"PC1 before={pc1_before:.1f}% -> after cf-adjustment={pc1_after:.1f}%  "
          f"(mean|r| {mr_before:.3f} -> {mr_after:.3f})")

    resid_df = pd.DataFrame(resid, columns=list(range(1, 51)), index=shares.index)
    resid_df.to_csv(OUT / f"daily_resid_llm_{lang}.csv")

    # ---- TY all pairs + FDR on residual series ----
    nodes = [s for s in range(1, 51) if resid_df[s].nunique() > 1]
    orders = {s: integration_order(resid_df[s]) for s in nodes}
    rows = []
    for i, a in enumerate(nodes):
        for b in nodes:
            if a == b:
                continue
            pair = resid_df[[a, b]].copy()
            pair.columns = ["source", "target"]
            p_lag = choose_var_lag(pair, MAX_LAG)
            dmax = max(orders[a], orders[b])
            pv = ty_wald_pvalue(pair, "source", "target", p_lag, dmax)
            if not np.isfinite(pv):
                continue
            lead, r = best_positive_lead_corr(resid_df[a], resid_df[b])
            rows.append((a, b, p_lag, dmax, lead, pv, r, abs(r)))
        if (i + 1) % 10 == 0:
            print(f"  ...source nodes done {i+1}/{len(nodes)}  ({time.time()-t0:.0f}s)")
    edges = pd.DataFrame(rows, columns=["source", "target", "p_lag", "dmax",
                                        "lead_days", "ty_pvalue", "pearson_r", "abs_r"])
    edges["fdr_significant"] = bh_fdr(edges["ty_pvalue"].values, FDR_Q)
    edges.insert(0, "language", lang)
    edges.to_csv(OUT / f"ty_edges_allpairs_cfadj_{lang}.csv", index=False)

    n_tested = len(edges)
    n_raw05 = int((edges["ty_pvalue"] < 0.05).sum())
    n_fdr = int(edges["fdr_significant"].sum())
    fdr_edges = edges[edges["fdr_significant"]]
    print(f"tested={n_tested}  p<.05(raw)={n_raw05}  BH-FDR(5%)={n_fdr}  "
          f"elapsed={time.time()-t0:.0f}s")
    summary_rows.append({
        "language": lang, "PC1_before": round(pc1_before, 1), "PC1_after": round(pc1_after, 1),
        "mean_abs_r_before": round(mr_before, 3), "mean_abs_r_after": round(mr_after, 3),
        "pairs_tested": n_tested, "raw_p05": n_raw05, "fdr_5pct": n_fdr,
        "fdr_mean_abs_r": round(float(fdr_edges["abs_r"].mean()), 3) if n_fdr else np.nan,
        "fdr_median_lead": float(fdr_edges["lead_days"].median()) if n_fdr else np.nan,
    })

summ = pd.DataFrame(summary_rows)
summ.to_csv(OUT / "summary_v5c_cfadj.csv", index=False)
print("\n===== SUMMARY (common-factor-adjusted) =====")
print(summ.to_string(index=False))
print("\nDONE ->", OUT)

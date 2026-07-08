# -*- coding: utf-8 -*-
"""Phase 1+2 of the v3 plan.

Phase 1: rebuild daily sectoral series per language with
  (a) uniform low-evidence rows excluded, (b) within-day share transform.
  Also recompute aggregate exposure V_s and negative-hostility H_s
  excluding uniform rows (new Table 2 candidate).

Phase 2: re-run pairwise Toda-Yamamoto tests on the corrected series,
  collect ALL p-values (no pre-filter), apply Benjamini-Hochberg FDR (5%),
  annotate with best-lead lagged Pearson correlations.

Outputs -> figures/causal network_v4_share_fdr/
Reuses the TY machinery from the maintained network script.
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

OUT = BASE / "figures" / "causal network_v4_share_fdr"
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
    """Return boolean mask of BH-FDR discoveries."""
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


summary_rows = []
indicator_frames = []

for lang, fname in LANGS.items():
    t0 = time.time()
    print(f"=== {lang} ===")
    df = pd.read_csv(BASE / fname,
                     usecols=["published_at", "negativity", "war_related"] + PROB,
                     low_memory=False)
    dates = pd.to_datetime(df["published_at"], errors="coerce", dayfirst=True).dt.floor("D")
    ok = dates.notna()
    df, dates = df[ok], dates[ok]
    P = df[PROB].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    neg = pd.to_numeric(df["negativity"], errors="coerce").fillna(0.0)
    war = pd.to_numeric(df["war_related"], errors="coerce").fillna(0.0)

    uniform = (P.max(axis=1) - P.min(axis=1)) < 1e-9
    keep = ~uniform
    print(f"articles={len(df)}, uniform excluded={uniform.sum()} ({uniform.mean()*100:.1f}%)")

    # ---- Phase 1a: aggregate indicators excluding uniform rows ----
    V = P[keep].sum(axis=0)
    H = P[keep].mul(neg[keep] * war[keep], axis=0).sum(axis=0)
    ind = pd.DataFrame({
        "V1": range(1, 51),
        "V_total": V.values, "H_neg_hostility": H.values,
        "V_share": (V / V.sum()).values, "H_share": (H / H.sum()).values,
    })
    ind.insert(0, "language", lang)
    ind.to_csv(OUT / f"indicators_exuniform_{lang}.csv", index=False)
    indicator_frames.append(ind)

    # ---- Phase 1b: corrected daily series (excl uniform + shares) ----
    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = P[keep].groupby(dates[keep].values).sum().reindex(idx, fill_value=0.0)
    daily.columns = list(range(1, 51))
    tot = daily.sum(axis=1)
    shares = daily.div(tot.replace(0, np.nan), axis=0).fillna(0.0)
    shares.index.name = "date"
    shares.to_csv(OUT / f"daily_share_exuniform_{lang}.csv")

    # PC1 diagnostic on the corrected series
    X = shares.values
    active = X.std(0) > 0
    Z = (X[:, active] - X[:, active].mean(0)) / X[:, active].std(0)
    C = np.corrcoef(Z.T)
    ev = np.linalg.eigvalsh(C)
    pc1 = ev.max() / ev.sum() * 100
    iu = np.triu_indices(C.shape[0], 1)
    mabsr = np.abs(C[iu]).mean()
    print(f"corrected series: PC1={pc1:.1f}%  mean|r|={mabsr:.3f}")

    # ---- Phase 2: TY over all ordered pairs, full p-values ----
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
            print(f"  ...source nodes done {i+1}/{len(nodes)}  ({time.time()-t0:.0f}s)")
    edges = pd.DataFrame(rows, columns=["source", "target", "p_lag", "dmax",
                                        "lead_days", "ty_pvalue", "pearson_r", "abs_r"])
    edges["fdr_significant"] = bh_fdr(edges["ty_pvalue"].values, FDR_Q)
    edges.insert(0, "language", lang)
    edges.to_csv(OUT / f"ty_edges_allpairs_{lang}.csv", index=False)

    n_tested = len(edges)
    n_raw05 = int((edges["ty_pvalue"] < 0.05).sum())
    n_fdr = int(edges["fdr_significant"].sum())
    fdr_edges = edges[edges["fdr_significant"]]
    print(f"tested={n_tested}  p<.05(raw)={n_raw05}  BH-FDR(5%)={n_fdr}  "
          f"elapsed={time.time()-t0:.0f}s")
    summary_rows.append({
        "language": lang, "articles": len(df), "uniform_excluded": int(uniform.sum()),
        "PC1_pct_corrected": round(pc1, 1), "mean_abs_r_corrected": round(mabsr, 3),
        "pairs_tested": n_tested, "raw_p05": n_raw05, "fdr_5pct": n_fdr,
        "fdr_mean_abs_r": round(float(fdr_edges["abs_r"].mean()), 3) if n_fdr else np.nan,
        "fdr_median_lead": float(fdr_edges["lead_days"].median()) if n_fdr else np.nan,
    })

pd.concat(indicator_frames).to_csv(OUT / "indicators_exuniform_all.csv", index=False)
summ = pd.DataFrame(summary_rows)
summ.to_csv(OUT / "summary_v4_share_fdr.csv", index=False)
print("\n===== SUMMARY =====")
print(summ.to_string(index=False))
print("\nDONE ->", OUT)

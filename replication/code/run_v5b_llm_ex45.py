# -*- coding: utf-8 -*-
"""Diagnostic: does excluding the public-administration catch-all sector
(V1=45) from the within-day share renormalization remove the compositional
common factor introduced by LLM classification's heavy V1=45 fallback mass?

Recomputes daily shares over the remaining 49 sectors only (renormalized to
sum 1 excluding V1=45), then reports PC1 / mean|r| for comparison against
run_v5_llm_indicators_fdr.py's full-50-sector result.
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
LLM_DIR = BASE / "figures" / "llm_full"
PROB = [f"IO_V1_{i}" for i in range(1, 51)]
LANGS = {
    "arabic": "iran_ar_v3_with_io_matrix.csv",
    "chinese": "iran_ch_v3_with_io_matrix.csv",
    "english": "iran_en_v3_with_io_matrix.csv",
    "persian": "iran_pe_v3_with_io_matrix.csv",
}

for lang, fname in LANGS.items():
    meta = pd.read_csv(BASE / fname, usecols=["published_at"], low_memory=False)
    meta["published_at"] = pd.to_datetime(meta["published_at"], errors="coerce", dayfirst=True).dt.floor("D")
    llm = pd.read_csv(LLM_DIR / f"llm_probs_full_{lang}.csv").set_index("orig_index")
    df = meta.join(llm, how="inner")
    dates = df["published_at"]
    ok = dates.notna()
    df, dates = df[ok], dates[ok]
    P = df[PROB].copy()

    mean_45 = P["IO_V1_45"].mean()

    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    daily_all = P.groupby(dates.values).sum().reindex(idx, fill_value=0.0)
    daily_all.columns = list(range(1, 51))

    # full-50 share (baseline, for reference)
    tot50 = daily_all.sum(axis=1)
    sh50 = daily_all.div(tot50.replace(0, np.nan), axis=1 if False else 0).fillna(0.0)

    # 49-sector share excluding V1=45, renormalized
    daily_49 = daily_all.drop(columns=45)
    tot49 = daily_49.sum(axis=1)
    sh49 = daily_49.div(tot49.replace(0, np.nan), axis=0).fillna(0.0)

    def pc1_meanr(sh):
        X = sh.values
        active = X.std(0) > 0
        Z = (X[:, active] - X[:, active].mean(0)) / X[:, active].std(0)
        C = np.corrcoef(Z.T)
        ev = np.linalg.eigvalsh(C)
        pc1 = ev.max() / ev.sum() * 100
        iu = np.triu_indices(C.shape[0], 1)
        return pc1, np.abs(C[iu]).mean()

    pc1_50, mr_50 = pc1_meanr(sh50)
    pc1_49, mr_49 = pc1_meanr(sh49)
    print(f"{lang:9s} mean(V1_45 weight)={mean_45:.3f}  "
          f"| 50-sector PC1={pc1_50:5.1f}% r={mr_50:.3f}  "
          f"| 49-sector(ex-45) PC1={pc1_49:5.1f}% r={mr_49:.3f}")

# -*- coding: utf-8 -*-
"""Build data/v5/<language>.json for the interactive viewer from the
replication package (share-transformed series, BH-FDR edge selection).

Edge row format (see meta.edge_format):
    [source_v1, target_v1, fdr_significant, ty_pvalue, ty_lag, lead_days, pearson_r]

Run from the repository root:
    python replication/code/build_viewer_v5.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "replication" / "data"
OUT = ROOT / "data" / "v5"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["arabic", "chinese", "english", "persian"]
lookup = pd.read_csv(DATA / "reference" / "sector_lookup_icio_v1_isic_rev4.csv", encoding="utf-8-sig")
lookup["V1"] = lookup["V1"].astype(int)

for lang in LANGS:
    ind = pd.read_csv(DATA / "daily_series" / f"daily_indicators_lexical_{lang}.csv")
    share = pd.read_csv(DATA / "daily_series" / f"daily_share_lexical_{lang}.csv", index_col=0)
    edges = pd.read_csv(DATA / "networks" / f"ty_edges_lexical_{lang}.csv")

    totals = dict(zip(ind["V1"].astype(int), ind["V_total"]))
    active = {int(c): int((share[c] > 0).sum()) for c in share.columns}
    nodes = [
        {"v1": int(r.V1), "total": round(float(totals.get(int(r.V1), 0.0)), 6),
         "active": active.get(int(r.V1), 0), "code": r.Code, "industry": r.Industry}
        for r in lookup.itertuples(index=False)
    ]
    rows = [
        [int(e.source), int(e.target), int(bool(e.fdr_significant)),
         round(float(e.ty_pvalue), 6), int(e.p_lag), int(e.lead_days), round(float(e.pearson_r), 4)]
        for e in edges.itertuples(index=False)
    ]
    payload = {
        "meta": {
            "dataset": "v5 (share-transformed series, BH-FDR 5%)",
            "edge_format": ["s", "t", "fdr", "p", "pLag", "lead", "r"],
            "note": "specification of record; supersedes data/v3",
        },
        "nodes": nodes,
        "edges": rows,
    }
    path = OUT / f"{lang}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    n_fdr = sum(r[2] for r in rows)
    print(f"{lang}: {len(rows)} pairs, {n_fdr} FDR edges -> {path.relative_to(ROOT)}")

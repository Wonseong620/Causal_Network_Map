# -*- coding: utf-8 -*-
"""Regenerate the two aggregate-composition manuscript figures from the
semantic (LLM) classification, per the bifurcated specification:
  fig01_llm_heatmap.png  - language-sector integrated shock share (Fig 1)
  fig02_llm_weekly.png   - weekly top-sector shock trends (Fig 2)
Layouts mirror create_io_matrix_figures.py (fig01 / fig04) so the manuscript
figures keep their look; only the classifier input changes.
"""
import io, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "manuscript_v5"
PROB = [f"IO_V1_{i}" for i in range(1, 51)]
LANG_FILES = {
    "arabic": "iran_ar_v3.csv", "chinese": "iran_ch_v3.csv",
    "english": "iran_en_v3.csv", "persian": "iran_pe_v3.csv",
}
MATLAB = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 100,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

lookup = pd.read_csv(BASE / "icio_sectors_with_isic_rev4_descriptions.csv")
lookup["V1"] = lookup["V1"].astype(int)

sector_rows, weekly_rows = [], []
for lang, fname in LANG_FILES.items():
    raw = pd.read_csv(BASE / fname, usecols=["published_at", "negativity", "war_related"],
                      low_memory=False)
    llm = pd.read_csv(BASE / "figures" / "llm_full" / f"llm_probs_full_{lang}.csv")
    idx = llm["orig_index"].to_numpy(dtype=int)
    probs = llm[PROB].to_numpy(dtype=float)
    neg = pd.to_numeric(raw["negativity"], errors="coerce").fillna(0).to_numpy()[idx]
    war = pd.to_numeric(raw["war_related"], errors="coerce").fillna(0).to_numpy()[idx]
    week = pd.to_datetime(raw["published_at"], errors="coerce", dayfirst=True)\
             .dt.to_period("W").dt.start_time.to_numpy()[idx]
    w = neg * war
    shock = probs * w[:, None]
    tot = shock.sum(axis=0)
    for k in range(50):
        sector_rows.append({"language": lang, "V1": k + 1, "shock": tot[k]})
    wk = pd.DataFrame(shock, columns=range(1, 51))
    wk["week"] = week
    m = wk.groupby("week").sum()
    for v1 in range(1, 51):
        for wdate, val in m[v1].items():
            weekly_rows.append({"language": lang, "V1": v1, "week": wdate, "shock": val})
    print(lang, "done, articles:", len(idx))

sector = pd.DataFrame(sector_rows)
weekly = pd.DataFrame(weekly_rows)

# ---------- Fig 1: heatmap ----------
top20 = sector.groupby("V1")["shock"].sum().sort_values(ascending=False).head(20).index.tolist()
pivot = (sector[sector["V1"].isin(top20)]
         .pivot(index="language", columns="V1", values="shock")
         .reindex(index=list(LANG_FILES), columns=top20))
row_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(14, 5.4))
im = ax.imshow(row_norm.to_numpy(), aspect="auto", cmap="viridis")
ax.set_title("Language-Sector Integrated Shock Share — semantic (LLM) classifier")
ax.set_xlabel("ICIO sector, V1 number")
ax.set_ylabel("Language")
ax.set_xticks(range(len(top20))); ax.set_xticklabels(top20)
ax.set_yticks(range(4)); ax.set_yticklabels([l.capitalize() for l in LANG_FILES])
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Within-language shock share")
fig.text(0.01, 0.006,
         "Note: Shock_s,l = sum_i(p_is x negativity_i x war_related_i) under the semantic (LLM) "
         "classifier, normalized within language. Top 20 sectors by pooled shock shown.",
         fontsize=9, color="0.25")
fig.subplots_adjust(bottom=0.2)
fig.savefig(OUT / "fig01_llm_heatmap.png")
plt.close(fig)
print("saved fig01_llm_heatmap.png | top20:", top20)

# ---------- Fig 2: weekly trends ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
legend_rows = []
for ax, lang in zip(axes.ravel(), LANG_FILES):
    top5 = (sector[sector["language"].eq(lang)]
            .sort_values("shock", ascending=False).head(5)["V1"].tolist())
    sub = weekly[weekly["language"].eq(lang) & weekly["V1"].isin(top5)]
    for i, v1 in enumerate(top5):
        line = sub[sub["V1"].eq(v1)].sort_values("week")
        ax.plot(line["week"], line["shock"], color=MATLAB[i % 7], linewidth=1.6,
                label=f"V1 {v1}")
        ind = lookup.loc[lookup["V1"] == v1, "Industry"].iloc[0]
        legend_rows.append({"language": lang, "V1": v1, "Industry": ind})
    ax.set_title(lang.capitalize())
    ax.set_ylabel("Weekly integrated shock")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
fig.suptitle("Weekly Sector Shock Trends, Top 5 Sectors by Language — semantic (LLM) classifier",
             y=0.985, fontsize=14, fontweight="bold")
fig.text(0.01, 0.005,
         "Note: Weekly shock sums the LLM sector probability weighted by negativity and "
         "war_related. Legend uses V1 numbers; see fig02_llm_weekly_legend.csv for names.",
         fontsize=9, color="0.25")
fig.subplots_adjust(bottom=0.12, hspace=0.4, wspace=0.22, right=0.96, top=0.9)
fig.savefig(OUT / "fig02_llm_weekly.png")
plt.close(fig)
pd.DataFrame(legend_rows).to_csv(OUT / "fig02_llm_weekly_legend.csv", index=False, encoding="utf-8-sig")
print("saved fig02_llm_weekly.png")

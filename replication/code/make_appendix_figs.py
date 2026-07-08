# -*- coding: utf-8 -*-
"""Appendix figures:
  fig12_mapping_rule.png       - decision hierarchy of the sector mapping rule (Appendix A)
  fig13_classifier_agreement.png - lexical vs semantic agreement, by language (Appendix B)
Requires agree_cos_{lang}.npy / agreement_stats.json in figures/llm_full/
(produced by the per-article agreement computation).
"""
import io, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "manuscript_v5"
LLM = BASE / "figures" / "llm_full"

# ============================================================
# Fig 12: mapping-rule decision hierarchy
# ============================================================
TEAL = dict(fc="#E1F5EE", ec="#0F6E56", tc="#085041")
BLUE = dict(fc="#E6F1FB", ec="#185FA5", tc="#0C447C")
GRAY = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")
AMBER = dict(fc="#FAEEDA", ec="#854F0B", tc="#412402")

fig, ax = plt.subplots(figsize=(9.2, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

def box(x, y, w, h, title, sub, style, fs=11.5, sfs=9.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                        linewidth=1.1, facecolor=style["fc"], edgecolor=style["ec"], zorder=2)
    ax.add_patch(b)
    if sub:
        ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=fs,
                 fontweight="bold", color=style["tc"], zorder=3)
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center", fontsize=sfs,
                 color=style["tc"], zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fs,
                 fontweight="bold", color=style["tc"], zorder=3)

def arrow(x1, y1, x2, y2, label=None, color="#5F5E5A"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                         color=color, linewidth=1.3, shrinkA=1, shrinkB=1, zorder=1)
    ax.add_patch(a)
    if label:
        if abs(y2 - y1) < 0.5:  # horizontal arrow: label above midpoint
            ax.text((x1+x2)/2, y1 + 1.2, label, fontsize=9.5, color=color,
                     ha="center", va="bottom", fontstyle="italic")
        else:  # vertical arrow: label to the right
            ax.text((x1+x2)/2 + 1.6, (y1+y2)/2, label, fontsize=9.5, color=color,
                     ha="left", va="center", fontstyle="italic")

box(31, 63, 38, 6, "Article text fields", "subtopic, title, factual summary, opinion, body", GRAY)
QX, QW = 6, 46
OX, OW = 60, 36
box(QX, 50, QW, 8, "Tier 1 — Direct industry evidence?",
    "oil fields, refineries, tankers, ports, banks,\nsanctioned payments, hospitals, telecom, aircraft", TEAL)
box(OX, 50, OW, 8, "Positive weights on\nmatched sectors",
    "main correspondences in Table A1", BLUE)
box(QX, 37, QW, 8, "Tier 2 — Explicit economic subtopic?",
    "energy, trade, finance, food security,\ninfrastructure and similar subtopic labels", TEAL)
box(OX, 37, OW, 8, "Weights on subtopic sectors", "", BLUE)
box(QX, 24, QW, 8, "Tier 3 — Mainly diplomatic, military,\nor administrative?",
    "no specific affected industry named", TEAL)
box(OX, 24, OW, 8, "Mass to public administration\nand defence (V1 = 45)", "", BLUE)
box(QX, 11, QW, 8, "Tier 4 — No usable economic evidence",
    "purely political salience, unusable text", TEAL)
box(OX, 11, OW, 8, "Uniform 0.02 on all 50 sectors",
    "flagged low-evidence; excluded from daily series", AMBER)

arrow(50, 63, 29, 58.3)
arrow(29, 50, 29, 45.3, "no")
arrow(29, 37, 29, 32.3, "no")
arrow(29, 24, 29, 19.3, "no")
for y in (54, 41, 28, 15):
    arrow(QX+QW, y, OX, y, "yes" if y != 15 else None)

box(18, 1.5, 64, 5.5, "Article-level exposure vector over 50 ICIO V1 sectors",
    "weights clipped to [0, 1] and normalized to sum to one", GRAY)
arrow(78, 11, 62, 7.2)
ax.text(50, 68.7, "", fontsize=1)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig12_mapping_rule.png", dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("saved fig12_mapping_rule.png")

# ============================================================
# Fig 13: classifier agreement by language
# ============================================================
stats = json.load(open(LLM / "agreement_stats.json"))
LANGS = ["arabic", "chinese", "english", "persian"]
LAB = {"arabic": "Arabic", "chinese": "Chinese", "english": "English", "persian": "Persian"}
cos = {l: np.load(LLM / f"agree_cos_{l}.npy") for l in LANGS}

C0, C1 = "#0072BD", "#D95319"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

bp = ax1.boxplot([cos[l] for l in LANGS], tick_labels=[LAB[l] for l in LANGS],
                 showfliers=False, patch_artist=True, widths=0.55,
                 medianprops=dict(color="black", linewidth=1.4))
for patch in bp["boxes"]:
    patch.set_facecolor("#B5D4F4"); patch.set_edgecolor(C0)
pooled_mean = sum(stats[l]["cos_mean"] * stats[l]["n"] for l in LANGS) / sum(stats[l]["n"] for l in LANGS)
ax1.axhline(pooled_mean, color=C1, linestyle="--", linewidth=1.3)
ax1.text(4.42, pooled_mean + 0.015, f"pooled mean {pooled_mean:.2f}", color=C1,
          fontsize=10, ha="right")
ax1.set_ylabel("Per-article cosine similarity", fontsize=11)
ax1.set_title("(a) Cosine similarity between the two classifiers'\n50-sector exposure vectors", fontsize=12, fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_ylim(0, 1)

x = np.arange(len(LANGS))
all_v = [stats[l]["top1"] for l in LANGS]
ex_v = [stats[l]["top1_exuni"] for l in LANGS]
n_non = [stats[l]["n"] * (1 - stats[l]["uni_pct"]/100) for l in LANGS]
pooled_all = sum(stats[l]["top1"] * stats[l]["n"] for l in LANGS) / sum(stats[l]["n"] for l in LANGS)
pooled_ex = sum(e * n for e, n in zip(ex_v, n_non)) / sum(n_non)
w = 0.36
ax2.bar(x - w/2, all_v, w, color=C0, label="All articles")
ax2.bar(x + w/2, ex_v, w, color=C1, label="Excluding lexical-uniform articles")
for xi, (a, e) in enumerate(zip(all_v, ex_v)):
    ax2.text(xi - w/2, a + 0.7, f"{a:.0f}", ha="center", fontsize=9.5)
    ax2.text(xi + w/2, e + 0.7, f"{e:.0f}", ha="center", fontsize=9.5)
ax2.set_xticks(x); ax2.set_xticklabels([LAB[l] for l in LANGS], fontsize=11)
ax2.set_ylabel("Top-sector agreement (%)", fontsize=11)
ax2.set_title(f"(b) Top-sector agreement\n(pooled: {pooled_all:.1f}% all, {pooled_ex:.1f}% excluding uniform)",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=10, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig13_classifier_agreement.png", dpi=220, facecolor="white")
plt.close(fig)
print("saved fig13_classifier_agreement.png")
print(f"pooled: cos={pooled_mean:.3f} top1_all={pooled_all:.1f} top1_ex={pooled_ex:.1f}")

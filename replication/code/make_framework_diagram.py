# -*- coding: utf-8 -*-
"""Render the shock-mapping methodology framework as a static PNG (matplotlib).
Coordinates are in inches (axes span the full figure), so box sizes are
measured directly against rendered text extents -- no guessing character
widths in data units.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"G:\My Drive\WSK.GDRIVE\QQ_shock network\IO table\figures\manuscript_v5\fig_framework_overview.png"

BLUE = dict(fc="#E6F1FB", ec="#185FA5", tc="#0C447C")
TEAL = dict(fc="#E1F5EE", ec="#0F6E56", tc="#085041")
GRAY = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")
CORAL = dict(fc="#FAECE7", ec="#993C1D", tc="#4A1B0C")

FIG_W, FIG_H = 9.6, 10.6
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

renderer = fig.canvas.get_renderer()

def text_w(text, fontsize, weight="normal"):
    t = ax.text(0, -100, text, fontsize=fontsize, fontweight=weight)
    bbox = t.get_window_extent(renderer=renderer)
    w_in = bbox.width / fig.dpi
    t.remove()
    return w_in

def box(cx, top_y, title, subtitle_lines, style, title_fs=12.5, sub_fs=9.5,
        min_w=2.0, pad_x=0.22, pad_top=0.14, pad_between=0.10, line_gap=0.19,
        pad_bottom=0.14):
    title_w = text_w(title, title_fs, "bold")
    sub_w = max([text_w(s, sub_fs) for s in subtitle_lines], default=0)
    w = max(min_w, title_w, sub_w) + 2 * pad_x
    title_h = title_fs / 72 * 1.25
    sub_h = len(subtitle_lines) * line_gap
    h = pad_top + title_h + pad_between + sub_h + pad_bottom
    x = cx - w / 2
    y = top_y - h
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                        linewidth=1.1, facecolor=style["fc"], edgecolor=style["ec"], zorder=2)
    ax.add_patch(b)
    ax.text(cx, top_y - pad_top - title_h * 0.55, title, ha="center", va="center",
             fontsize=title_fs, color=style["tc"], fontweight="bold", zorder=3)
    sub_top = top_y - pad_top - title_h - pad_between
    for i, s in enumerate(subtitle_lines):
        ax.text(cx, sub_top - line_gap * (i + 0.55), s, ha="center", va="center",
                 fontsize=sub_fs, color=style["tc"], zorder=3)
    return dict(cx=cx, x=x, y=y, w=w, h=h, top=top_y, bottom=y, left=x, right=x + w)

def varrow(b1, b2, color="#5F5E5A", lw=1.3, ls="-"):
    x1 = b1["cx"]; y1 = b1["bottom"]
    x2 = b2["cx"]; y2 = b2["top"]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                         color=color, linewidth=lw, linestyle=ls, shrinkA=1, shrinkB=1, zorder=1)
    ax.add_patch(a)

def harrow(b1, b2, color="#888780", lw=1.1, ls="--"):
    y = (b1["top"] + b1["bottom"]) / 2
    a = FancyArrowPatch((b1["right"], y), (b2["left"], y), arrowstyle="-|>", mutation_scale=9,
                         color=color, linewidth=lw, linestyle=ls, shrinkA=1, shrinkB=1, zorder=1)
    ax.add_patch(a)

def midlabel(b1, b2, text, fontsize=8.3, color="#5F5E5A", dy=0.0):
    x = b1["cx"]
    y = (b1["bottom"] + b2["top"]) / 2 + dy
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)

def diag_arrow(bA, bB, color="#5F5E5A", lw=1.3, rad=0.15):
    a = FancyArrowPatch((bA["cx"], bA["bottom"]), (bB["cx"], bB["top"]), arrowstyle="-|>",
                         mutation_scale=11, color=color, linewidth=lw,
                         connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)

# layout: two columns
LX, RX = FIG_W * 0.27, FIG_W * 0.73
TOP = FIG_H - 0.35

corpus = box(FIG_W / 2, TOP, "Multilingual news corpus",
             ["58,530 articles · Arabic, Chinese, English, Persian"], GRAY, min_w=4.6)

y1 = corpus["bottom"] - 0.55
lex = box(LX, y1, "Lexical classification", ["TF-IDF, 50-sector cosine similarity"], TEAL)
sem = box(RX, y1, "Semantic classification", ["Claude Haiku 4.5, zero-shot JSON"], BLUE)
varrow(corpus, lex)
varrow(corpus, sem)

y2 = min(lex["bottom"], sem["bottom"]) - 0.5
share = box(LX, y2, "Daily share series", ["X_slt / Σs X_slt, by language and sector"], TEAL)
pooled = box(RX, y2, "Pooled composition shares", ["aggregate exposure share, all articles"], BLUE)
varrow(lex, share)
varrow(sem, pooled)

y3 = min(share["bottom"], pooled["bottom"]) - 0.5
net = box(LX, y3, "Dynamic network estimation",
          ["ADF → Toda-Yamamoto lead-lag test", "→ BH-FDR (q=0.05), all sector pairs"], TEAL,
          min_w=3.1)
comp = box(RX, y3, "Composition indicators",
           ["Table 2b sector ranking,", "negative-hostility exposure by language"], BLUE, min_w=3.1)
varrow(share, net)
varrow(pooled, comp)
harrow(net, comp)
ax.text((net["right"] + comp["left"]) / 2, (net["top"] + net["bottom"]) / 2 + 0.22,
         "robustness\ncomparison", ha="center", va="center", fontsize=8, color="#888780")

y4 = net["bottom"] - 0.5
io = box(LX, y4, "Input-output structural validation",
         ["Fisher exact test + QAP correlation vs.", "OECD ICIO 50×50 coefficient matrix A"],
         CORAL, min_w=3.5)
varrow(net, io)

y5 = min(io["bottom"], comp["bottom"] - 0.4) - 0.55
report = box(FIG_W / 2, y5, "Manuscript reporting",
             ["sectoral indicators, dynamic network, IO-alignment evidence,",
              "and the common-factor limitation — inputs for production-network models"],
             GRAY, min_w=6.6)
diag_arrow(io, report, rad=-0.25)
diag_arrow(comp, report, rad=0.25)

# caption footer
ax.text(FIG_W / 2, report["bottom"] - 0.32,
         "Left column = §6–7 dynamic-network specification (TF-IDF).   "
         "Right column = §3.1 composition specification (LLM).",
         ha="center", va="center", fontsize=8.6, color="#5F5E5A", style="italic")

plt.savefig(OUT, dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved", OUT)

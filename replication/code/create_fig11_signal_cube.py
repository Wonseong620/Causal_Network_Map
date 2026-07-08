# -*- coding: utf-8 -*-
"""Fig 11 / Appendix C — Signal Cube: 3D Layered View.

Pseudo-3D stacked-glass-layer figure decomposing sectoral lead-lag signals
by polarity (positive / negative) and expression explicitness (direct =
sector explicitly named in text; indirect = implied via contextual economic
cues). Pure matplotlib; outputs 300-dpi PNG + PDF to figures/.
"""
import io
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------- palette ----------------
TEAL = "#0e8f8f"
TEAL_LT = "#39a8a8"
ORANGE = "#e2571f"
ORANGE_LT = "#ec8050"
GREEN = "#3f9e3f"
INK = "#1a1a1a"
MUTED = "#6b7280"
PASTELS = ["#f4a9a0", "#a8c9ef", "#f5c98a", "#b8dfae", "#d5b8e8",
           "#9fd8df", "#f3b8d0", "#c9c3ef", "#ffd9a8", "#b5e3c9"]

plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})

fig = plt.figure(figsize=(15.0, 11.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1500)
ax.set_ylim(0, 1120)
ax.axis("off")

# ---------------- title ----------------
ax.text(700, 1078, "Appendix C. Signal Cube: 3D Layered View",
        ha="center", va="center", fontsize=27, fontweight="bold", color=INK)
ax.text(700, 1040, "Decomposition by tone and risk order — 18,013 risk-annotated articles, full LLM classification",
        ha="center", va="center", fontsize=15.5, style="italic", color=MUTED)

# ---------------- layer geometry ----------------
CX = 540
RX, RY = 205, 72
PLANE_HW, PLANE_HD = 330, 118
SHEAR = 95
LAYER_Y = [880, 655, 430, 205]

LAYERS = [
    dict(fill="#e4f4f2", edge="#bcdcd8", color=TEAL, dash=False,
         label="Positive-direct",
         desc="Low-negativity tone,\nfirst-order (direct)\nrisk exposure\nn = 5,644",
         nodes=[("pa", 95, "#7c4fb0", "Public admin & defence", (0, 26)),
                ("oil", 148, "#3a7fd5", "Crude oil & gas", (-72, 16)),
                ("wt", 32, "#189898", "Water transport", (72, 14)),
                ("wh", 185, "#f08c1e", "Warehousing & support", (-92, 16))],
         widths=[4.2, 3.4, 2.7, 2.0], cues=None),
    dict(fill="#eaf6f4", edge="#c9e4e0", color=TEAL_LT, dash=True,
         label="Positive-indirect",
         desc="Low-negativity tone,\nsecond-order (spillover)\nrisk exposure\nn = 3,349",
         nodes=[("pa", 95, "#7c4fb0", "Public admin & defence", (0, 26)),
                ("oil", 150, "#3a7fd5", "Crude oil & gas", (-72, 16)),
                ("wt", 32, "#189898", "Water transport", (72, 14)),
                ("fin", 187, "#d4a017", "Finance & insurance", (-88, 14))],
         widths=[3.6, 3.2, 2.4, 2.4],
         cues=[("oil / energy", -150, 22), ("transport / shipping", 10, 44), ("currency / banking", 158, 8)]),
    dict(fill="#fdeee6", edge="#f2d4c4", color=ORANGE, dash=False,
         label="Negative-direct",
         desc="High-negativity tone,\nfirst-order (direct)\nrisk exposure\nn = 6,208",
         nodes=[("pa", 95, "#7c4fb0", "Public admin & defence", (0, 26)),
                ("oil", 148, "#3a7fd5", "Crude oil & gas", (-72, 16)),
                ("wt", 32, "#189898", "Water transport", (72, 14)),
                ("wh", 185, "#f08c1e", "Warehousing & support", (-92, 16))],
         widths=[4.3, 3.3, 2.7, 1.9], cues=None),
    dict(fill="#fdf3ec", edge="#f4ded0", color=ORANGE_LT, dash=True,
         label="Negative-indirect",
         desc="High-negativity tone,\nsecond-order (spillover)\nrisk exposure\nn = 2,812",
         nodes=[("pa", 95, "#7c4fb0", "Public admin & defence", (0, 26)),
                ("oil", 150, "#3a7fd5", "Crude oil & gas", (-72, 16)),
                ("wt", 32, "#189898", "Water transport", (72, 14)),
                ("fin", 187, "#d4a017", "Finance & insurance", (-88, 14))],
         widths=[3.9, 3.5, 2.5, 2.0],
         cues=[("oil / energy", -150, 22), ("supply chain", 10, 44), ("transport / shipping", 158, 8)]),
]


# top -> bottom: Negative-direct, Negative-indirect, Positive-indirect,
# Positive-direct (negativity at the top of the cube, positivity at the base)
LAYERS = [LAYERS[2], LAYERS[3], LAYERS[1], LAYERS[0]]


def plane_corners(cy):
    return [(CX - PLANE_HW + SHEAR, cy + PLANE_HD),
            (CX + PLANE_HW + SHEAR, cy + PLANE_HD),
            (CX + PLANE_HW - SHEAR, cy - PLANE_HD),
            (CX - PLANE_HW - SHEAR, cy - PLANE_HD)]


def ring_pos(cy, angle_deg):
    a = np.radians(angle_deg)
    return CX + RX * np.cos(a), cy + RY * np.sin(a)


# faint vertical guide lines connecting layer corners (cube outline)
top_c = plane_corners(LAYER_Y[0])
bot_c = plane_corners(LAYER_Y[-1])
for (x1, y1), (x2, y2) in zip(top_c, bot_c):
    ax.add_line(Line2D([x1, x2], [y1, y2], color="#d9dee3", lw=0.9, zorder=0))
ax.add_patch(Polygon([(x, y + 8) for x, y in top_c], closed=True,
                     facecolor="none", edgecolor="#e2e6ea", lw=0.9, zorder=0))

rng = np.random.default_rng(7)

# draw bottom layer first so upper layers stack on top (built upward)
for stack_i, (cy, L) in enumerate(list(zip(LAYER_Y, LAYERS))[::-1]):
    zbase = stack_i * 10  # each higher layer sits above the one below
    # soft shadow + translucent plane
    ax.add_patch(Polygon([(x + 5, y - 7) for x, y in plane_corners(cy)],
                         closed=True, facecolor="#000000", alpha=0.045,
                         edgecolor="none", zorder=zbase + 1))
    ax.add_patch(Polygon(plane_corners(cy), closed=True, facecolor=L["fill"],
                         alpha=0.82, edgecolor=L["edge"], lw=1.2, zorder=zbase + 2))

    # de-emphasised pastel ring dots
    taken = [a for (_, a, _, _, _) in L["nodes"]] + [270]
    angles = [a for a in range(0, 360, 14)
              if min(abs((a - t + 180) % 360 - 180) for t in taken) > 9]
    for a in angles:
        x, y = ring_pos(cy, a)
        ax.add_patch(Circle((x, y), 6.5, facecolor=rng.choice(PASTELS),
                            edgecolor="white", lw=0.8, alpha=0.5, zorder=zbase + 3))

    fx, fy = ring_pos(cy, 270)
    dash = (0, (5, 3)) if L["dash"] else "solid"

    for (name, ang, ncol, lab, lab_off), lw in zip(L["nodes"], L["widths"]):
        nx, ny = ring_pos(cy, ang)
        rad = 0.22 if nx < fx else -0.22
        ax.add_patch(FancyArrowPatch(
            (fx, fy + 10), (nx, ny - 9),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_width=4.2,head_length=7.5",
            linestyle=dash, lw=lw, color=L["color"],
            shrinkA=2, shrinkB=4, zorder=zbase + 4))
        ax.add_patch(Circle((nx, ny), 9, facecolor=ncol, edgecolor="white",
                            lw=1.6, zorder=zbase + 5))
        if lab:
            ax.text(nx + lab_off[0], ny + lab_off[1], lab, ha="center",
                    va="center", fontsize=11.5, color=INK, zorder=zbase + 6)

    if L["cues"]:
        for text, dx, dy in L["cues"]:
            ax.text(fx + dx, fy + 55 + dy, text, ha="center", va="center",
                    fontsize=11, style="italic", color="#374151", zorder=zbase + 6)

    ax.add_patch(Circle((fx, fy), 11, facecolor=GREEN, edgecolor="white",
                        lw=2, zorder=zbase + 7))
    ax.text(fx, fy - 26, "Sector 26 = Other transport equipment",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="#2c6e2c", zorder=zbase + 7)

    # layer label pill + description
    pill_x, pill_w, pill_h = 905, 190, 40
    py = cy + 36
    ax.add_patch(FancyBboxPatch((pill_x, py - pill_h / 2), pill_w, pill_h,
                                boxstyle="round,pad=2,rounding_size=10",
                                facecolor="white", edgecolor=L["color"],
                                linestyle=(0, (4, 3)) if L["dash"] else "solid",
                                lw=1.8, zorder=100))
    ax.text(pill_x + pill_w / 2, py, L["label"], ha="center", va="center",
            fontsize=14.5, fontweight="bold", color=L["color"], zorder=101)
    ax.text(pill_x + pill_w / 2, py - 62, L["desc"], ha="center", va="center",
            fontsize=11, color="#374151", zorder=101, linespacing=1.35)

# ---------------- depth axis (anchored at the cube origin) ----------------
# Origin (0,0,0) = front-left corner of the BOTTOM layer; axis runs upward
# along the cube's left vertical edge.
ox, oy = plane_corners(LAYER_Y[-1])[3]          # bottom-front-left corner
tx = plane_corners(LAYER_Y[0])[3][1]            # top layer front-left height
ax.add_patch(FancyArrowPatch((ox, oy), (ox, tx + 130),
                             arrowstyle="-|>,head_width=4,head_length=8",
                             lw=1.8, color="#374151", zorder=50))
ax.text(ox, tx + 172, "Depth", ha="center", fontsize=14.5,
        fontweight="bold", color=INK, zorder=51)
ax.text(ox, tx + 152, "(decomposition layer)", ha="center", va="top",
        fontsize=11, color="#374151", zorder=51)
ax.text(ox - 14, tx + 60, "Negativity", ha="right", fontsize=12.5,
        fontweight="bold", color=ORANGE, zorder=51)
ax.text(ox - 14, tx + 38, "(harsher tone)", ha="right", va="top",
        fontsize=10.5, color=MUTED, linespacing=1.3, zorder=51)
ax.text(ox - 14, oy + 52, "Positivity", ha="right", fontsize=12.5,
        fontweight="bold", color=TEAL, zorder=51)
ax.text(ox - 14, oy + 30, "(milder tone)", ha="right", va="top",
        fontsize=10.5, color=MUTED, linespacing=1.3, zorder=51)

# ---------------- legend box (no heading; content fitted with margins) ----
LX, LW = 1140, 320
LEG_TOP, LEG_BOT = 985, 557
ax.add_patch(FancyBboxPatch((LX, LEG_BOT), LW, LEG_TOP - LEG_BOT,
                            boxstyle="round,pad=6,rounding_size=8",
                            facecolor="white", edgecolor="#c8cdd3", lw=1.2))

y = LEG_TOP - 28
ax.text(LX + 18, y, "Color = tone (vs language median)",
        fontsize=12, fontweight="bold", color=INK)
for lab, col in [("Low negativity", TEAL), ("High negativity", ORANGE)]:
    y -= 28
    ax.add_line(Line2D([LX + 26, LX + 72], [y, y], color=col, lw=3.2))
    ax.text(LX + 88, y, lab, fontsize=11, va="center", color="#374151")

y -= 40
ax.text(LX + 18, y, "Line style = risk order",
        fontsize=12, fontweight="bold", color=INK)
for lab, ls in [("First-order (direct exposure)", "solid"),
                ("Second-order (spillover)", (0, (5, 3)))]:
    y -= 28
    ax.add_line(Line2D([LX + 26, LX + 72], [y, y], color=INK, lw=2.2, linestyle=ls))
    ax.text(LX + 88, y, lab, fontsize=11, va="center", color="#374151")

y -= 40
ax.text(LX + 18, y, "Thickness = co-exposure share",
        fontsize=12, fontweight="bold", color=INK)
for lab, lw_ in [("0.21+", 4.2), ("0.10", 2.6), ("0.06", 1.6)]:
    y -= 26
    ax.add_line(Line2D([LX + 26, LX + 72], [y, y], color=INK, lw=lw_))
    ax.text(LX + 88, y, lab, fontsize=11, va="center", color="#374151")

y -= 40
ax.text(LX + 18, y, "Depth = decomposition layer",
        fontsize=12, fontweight="bold", color=INK)
for i, col in enumerate([ORANGE, ORANGE_LT, TEAL_LT, TEAL]):
    ax.add_line(Line2D([LX + 40, LX + 40], [y - 22 - i * 11, y - 33 - i * 11],
                       color=col, lw=4, solid_capstyle="butt"))
ax.annotate("", (LX + 40, y - 72), (LX + 40, y - 66),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))
ax.text(LX + 60, y - 30, "Top: negativity (harsher tone)",
        fontsize=10.5, va="center", color="#374151")
ax.text(LX + 60, y - 56, "Bottom: positivity (milder tone)",
        fontsize=10.5, va="center", color="#374151")

# ---------------- reading guide ----------------
ax.add_patch(FancyBboxPatch((LX, 168), LW, 300,
                            boxstyle="round,pad=6,rounding_size=8",
                            facecolor="white", edgecolor="#c8cdd3", lw=1.2))
ax.text(LX + 40, 441, "READING GUIDE", fontsize=13, fontweight="bold", color=INK)
guide = [
    "Each layer groups articles by tone\n(negativity below/above the language\nmedian) and annotated risk order.",
    "Edges show the top co-exposure sectors\nof Sector 26 within each layer, from the\nfull-corpus LLM sector probabilities.",
    "Edge thickness = share of the layer's\nco-exposure mass (top-4 sectors shown).",
    "Cue labels on dashed layers are the most\nfrequent economic subtopics among\nsecond-order articles.",
]
gy = 411
for g in guide:
    ax.text(LX + 24, gy, "•", fontsize=12, color=INK, va="top")
    ax.text(LX + 40, gy, g, fontsize=10.6, color="#374151", va="top", linespacing=1.3)
    gy -= 18 * (g.count("\n") + 1) + 12

# ---------------- footer ----------------
ax.text(700, 42,
        "Note: first/second order from the upstream risk_order annotation; tone split at the language-specific negativity median.\n"
        "Co-exposure of sector s with Sector 26 (other transport equipment): sum over articles of p(26) x p(s). Non-highlighted sectors are de-emphasized; top-4 sectors shown per layer.",
        ha="center", fontsize=11.5, style="italic", color=MUTED)

fig.savefig(OUT / "fig11_signal_cube_appendixC.png", dpi=300, facecolor="white")
try:
    fig.savefig(OUT / "fig11_signal_cube_appendixC.pdf", facecolor="white")
except PermissionError:
    print("WARNING: PDF is open in another program - PNG updated, PDF skipped.")
print("saved ->", OUT / "fig11_signal_cube_appendixC.png")

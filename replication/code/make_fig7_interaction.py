# -*- coding: utf-8 -*-
"""Fig 7: static recreation of the web viewer's three lead-lag interaction
modes (hover / lead-only / lag-only) for a focal sector, using the Section 6
specification of record (English, share-transformed, BH-FDR q=0.05).
Ego-network layout: focal sector at the centre, FDR partners around it.
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
FIGOUT = BASE / "figures" / "manuscript_v5"
V4_DIR = BASE / "figures" / "causal network_v4_share_fdr"
FOCAL = 20

lookup = pd.read_csv(BASE / "icio_sectors_with_isic_rev4_descriptions.csv")
lookup["V1"] = lookup["V1"].astype(int)
lab = dict(zip(lookup["V1"], lookup["Industry"]))

def short(v, maxlen=26):
    t = lab[v].split(";")[0].split(",")[0]
    if t.startswith("Manufacture of "):
        t = t[len("Manufacture of "):].capitalize()
    if len(t) > maxlen:
        t = t[:maxlen].rsplit(" ", 1)[0] + "…"
    return t

edges_all = pd.read_csv(V4_DIR / "ty_edges_allpairs_english.csv")
sig = edges_all[edges_all["fdr_significant"]]
ego = sig[(sig["source"] == FOCAL) | (sig["target"] == FOCAL)].reset_index(drop=True)
partners = sorted(set(ego["source"]) | set(ego["target"]) - {FOCAL})
partners = [p for p in partners if p != FOCAL]
print("partners:", partners)

# fixed positions: focal centre, partners spread on a circle
angles = np.linspace(90, 90 + 360, len(partners), endpoint=False)
pos = {FOCAL: (0.0, 0.0)}
for p, a in zip(partners, angles):
    pos[p] = (1.35 * np.cos(np.deg2rad(a)), 1.05 * np.sin(np.deg2rad(a)))

BLUE = "#0072BD"
ORANGE = "#D95319"
GRAY = "0.86"
GRAY_TXT = "0.62"

MODES = [
    ("Hover mode", "all incident edges", lambda u, v: True),
    ("Lead-only mode", f"sectors that V1={FOCAL} precedes", lambda u, v: u == FOCAL),
    ("Lag-only mode", f"sectors that precede V1={FOCAL}", lambda u, v: v == FOCAL),
]

NODE_R = 0.21

def name_label_pos(x, y):
    """Place the partner's name away from the centre, clear of the edges."""
    if abs(y) > abs(x):          # top / bottom node
        off = NODE_R + 0.14
        return x, y + np.sign(y) * off, "center", ("bottom" if y > 0 else "top")
    off = NODE_R + 0.10          # left / right node
    return x + np.sign(x) * off, y, ("left" if x > 0 else "right"), "center"

fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
for ax, (title, sub, keep) in zip(axes, MODES):
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}\n{sub}", fontsize=15.5, fontweight="bold", pad=12)
    for r in ego.itertuples(index=False):
        u, v = int(r.source), int(r.target)
        on = keep(u, v)
        x1, y1 = pos[u]; x2, y2 = pos[v]
        d = np.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        xa, ya = x1 + ux * NODE_R, y1 + uy * NODE_R
        xb, yb = x2 - ux * (NODE_R + 0.045), y2 - uy * (NODE_R + 0.045)
        col = BLUE if on else GRAY
        lw = (1.6 + 5.5 * r.abs_r) if on else 1.1
        a = FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                             mutation_scale=22 if on else 12, color=col, linewidth=lw,
                             alpha=0.95 if on else 0.65, zorder=2 if on else 1)
        ax.add_patch(a)
        if on:
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            # offset the label perpendicular to the edge so it never sits on it
            px, py = -uy, ux
            if px < 0 or (px == 0 and py < 0):
                px, py = -px, -py
            ax.text(mx + px * 0.22, my + py * 0.22, f"+{int(r.lead_days)}d, r={r.pearson_r:.2f}",
                     ha="center", va="center", fontsize=11.5, color=BLUE,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=3)
    for v in [FOCAL] + partners:
        x, y = pos[v]
        active = (v == FOCAL) or any(keep(int(r.source), int(r.target)) and v in (int(r.source), int(r.target))
                                     for r in ego.itertuples(index=False))
        if v == FOCAL:
            fc, ec, tc = ORANGE, "black", "white"
        elif active:
            fc, ec, tc = BLUE, "white", "white"
        else:
            fc, ec, tc = GRAY, "white", "white"
        ax.add_patch(Circle((x, y), NODE_R, facecolor=fc, edgecolor=ec,
                             linewidth=1.6 if v == FOCAL else 1.0, zorder=4))
        ax.text(x, y, str(v), ha="center", va="center", fontsize=13, fontweight="bold",
                 color=tc, zorder=5)
        lab_col = "0.25" if (active or v == FOCAL) else GRAY_TXT
        if v == FOCAL:
            lx, ly, ha, va = x, y - NODE_R - 0.14, "center", "top"
        else:
            lx, ly, ha, va = name_label_pos(x, y)
        ax.text(lx, ly, short(v), ha=ha, va=va, fontsize=10.5, color=lab_col, zorder=5)

fig.suptitle(f"Lead-lag interaction modes for V1={FOCAL} ({short(FOCAL, 40)}) — "
             "English network, BH-FDR 5%",
             fontsize=16, fontweight="bold", y=0.995)
fig.text(0.5, 0.035,
         "Static rendering of the interactive web viewer's three modes. Orange = selected sector; "
         "blue = highlighted partners; grey = de-emphasised.",
         ha="center", fontsize=11.5, color="0.3")
fig.text(0.5, 0.005,
         "Edge labels give the source sector's lead in days and the lagged Pearson correlation.",
         ha="center", fontsize=11.5, color="0.3")
fig.tight_layout(rect=[0, 0.06, 1, 0.92])
fig.savefig(FIGOUT / "fig7_interaction_modes.png", dpi=220)
print("saved", FIGOUT / "fig7_interaction_modes.png")

# -*- coding: utf-8 -*-
"""Generate six figures for manuscript v5:
  fig4_network_english.png   - English FDR network (TF-IDF share+FDR spec)
  fig5_network_4lang.png     - 2x2 panel, all four languages
  fig6_llm_top_sectors.png   - LLM-based top-8 sector bar chart (Table 2b)
  figB1_pc1_comparison.png   - PC1 across 4 pipeline stages, grouped by language
  figB2_rank_change.png      - TF-IDF vs LLM sector share dumbbell chart
  figB3_io_alignment.png     - FDR-edge rate: IO-adjacent vs non-adjacent pairs
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
FIGOUT = BASE / "figures" / "manuscript_v5"
FIGOUT.mkdir(parents=True, exist_ok=True)

TY_DIR = BASE / "figures" / "causal network_html_v4" / "html_generation_csv_and_code"
sys.path.insert(0, str(TY_DIR))
import create_ty_v1_networks as ty  # noqa: E402

LANGS = ["arabic", "chinese", "english", "persian"]
LANG_LABEL = {"arabic": "Arabic", "chinese": "Chinese", "english": "English", "persian": "Persian"}
lookup = pd.read_csv(BASE / "icio_sectors_with_isic_rev4_descriptions.csv")
lookup = lookup.rename(columns={"ISIC Rev.4": "ISIC_Rev4"})
lookup["V1"] = lookup["V1"].astype(int)
lab = dict(zip(lookup["V1"], lookup["Industry"]))

def short_label(v1, maxlen=30):
    text = lab[v1].split(";")[0].split(",")[0]
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen].rsplit(" ", 1)[0]
    return cut + "…"
V4_DIR = BASE / "figures" / "causal network_v4_share_fdr"
ty.set_matlab_style()

# ============================================================
# Fig 4 & 5: TF-IDF share+FDR networks
# ============================================================
def load_network(lang):
    daily = pd.read_csv(V4_DIR / f"daily_share_exuniform_{lang}.csv", index_col=0, parse_dates=True)
    daily.columns = [int(c) for c in daily.columns]
    nodes = ty.select_nodes(daily, lookup)
    edges_all = pd.read_csv(V4_DIR / f"ty_edges_allpairs_{lang}.csv")
    edges = edges_all[edges_all["fdr_significant"]].reset_index(drop=True)
    return nodes, edges

print("=== Fig 4: English network ===")
nodes_en, edges_en = load_network("english")

def draw_single_network(language, nodes, edges, output_path):
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["V1"].astype(int).tolist())
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), abs_r=float(row.abs_r))
    pos = ty.network_positions(nodes, edges)
    counts = nodes.set_index("V1")["total_occurrence"].to_dict()
    max_count = max(counts.values()) if counts else 1.0
    node_sizes = {int(v1): 120 + 520 * np.sqrt(float(counts[int(v1)]) / max_count) for v1 in graph.nodes}

    fig, ax = plt.subplots(figsize=(13.5, 13.5))
    ax.set_title(f"Toda-Yamamoto Lead-Lag Network: {language} V1 Sectors", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    edge_widths = [1.0 + 5.4 * graph[u][v]["abs_r"] for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=18,
                           width=edge_widths, edge_color=[ty.MATLAB_COLORS[0]] * graph.number_of_edges(),
                           alpha=0.44, connectionstyle="arc3,rad=0.10",
                           min_source_margin=7, min_target_margin=9)
    nodelist = list(graph.nodes)
    node_colors = [ty.MATLAB_COLORS[i % len(ty.MATLAB_COLORS)] for i, _ in enumerate(nodelist)]
    nx.draw_networkx_nodes(graph, pos, nodelist=nodelist, node_size=[node_sizes[v] for v in nodelist],
                           node_color=node_colors, edgecolors="white", linewidths=1.5, alpha=0.96, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={v: str(v) for v in graph.nodes}, font_color="white",
                            font_size=10, font_weight="bold", ax=ax)
    edge_labels = {(int(r.source), int(r.target)): f"+{int(r.lead_days)}d" for r in edges.itertuples(index=False)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8,
                                 font_color=ty.MATLAB_COLORS[0], rotate=False, label_pos=0.52,
                                 bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.58}, ax=ax)
    subtitle = f"50 V1 sectors | {len(edges)} BH-FDR-significant directed edges (q=0.05, share-transformed, uniform articles excluded, max lag=7d)"
    ax.text(0.5, 0.965, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=14.5, fontweight="bold")
    note = ("Node label = V1 number only. Node size proportional to sqrt(total daily soft occurrence). "
           "Arrow direction from Toda-Yamamoto Wald test; edge thickness proportional to |Pearson r|. "
           "Edges selected by Benjamini-Hochberg false discovery rate (5%) over all 2,450 ordered sector "
           "pairs, replacing the fixed |r| cutoff used in earlier versions.")
    fig.text(0.02, 0.015, "Note. " + note, ha="left", va="bottom", fontsize=11.5, color="0.25", wrap=True)
    legend_handles = [Line2D([0], [0], color=ty.MATLAB_COLORS[0], lw=5, label="BH-FDR-significant directed edge"),
                      Line2D([0], [0], marker="o", color="0.55", linestyle="None", markersize=14, label="Node size = occurrence")]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2, frameon=True, fontsize=14)
    fig.subplots_adjust(bottom=0.12, top=0.88)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

draw_single_network("English", nodes_en, edges_en, FIGOUT / "fig4_network_english.png")
print(f"  {len(edges_en)} edges -> fig4_network_english.png")

print("=== Fig 5: 4-language panel ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
for ax, lang in zip(axes.flat, LANGS):
    nodes, edges = load_network(lang)
    graph_pos = ty.network_positions(nodes, edges)
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(nodes["V1"].astype(int).tolist())
    for row in edges.itertuples(index=False):
        G.add_edge(int(row.source), int(row.target), abs_r=float(row.abs_r))
    counts = nodes.set_index("V1")["total_occurrence"].to_dict()
    max_count = max(counts.values()) if counts else 1.0
    sizes = {int(v1): 60 + 260 * np.sqrt(float(counts[int(v1)]) / max_count) for v1 in G.nodes}
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"{LANG_LABEL[lang]}  ({len(edges)} FDR edges)", fontsize=18, fontweight="bold")
    nx.draw_networkx_edges(G, graph_pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=10,
                           width=[1.0 + 3.0 * G[u][v]["abs_r"] for u, v in G.edges()],
                           edge_color=[ty.MATLAB_COLORS[0]] * G.number_of_edges(),
                           alpha=0.55, connectionstyle="arc3,rad=0.10",
                           min_source_margin=5, min_target_margin=7)
    nodelist = list(G.nodes)
    node_colors = [ty.MATLAB_COLORS[i % len(ty.MATLAB_COLORS)] for i, _ in enumerate(nodelist)]
    nx.draw_networkx_nodes(G, graph_pos, nodelist=nodelist, node_size=[sizes[v] for v in nodelist],
                           node_color=node_colors, edgecolors="white", linewidths=1.0, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(G, graph_pos, labels={v: str(v) for v in G.nodes}, font_color="white",
                            font_size=8, font_weight="bold", ax=ax)
fig.suptitle("Toda-Yamamoto lead-lag networks by language (share-transformed series, BH-FDR 5%)",
             fontsize=20, fontweight="bold", y=0.995)
legend_handles = [Line2D([0], [0], color=ty.MATLAB_COLORS[0], lw=4.5, label="FDR-significant directed edge"),
                  Line2D([0], [0], marker="o", color="0.55", linestyle="None", markersize=13,
                        label="Node size = occurrence")]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=True, fontsize=15, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.035, 1, 0.97])
fig.savefig(FIGOUT / "fig5_network_4lang.png", dpi=200)
plt.close(fig)
print("  saved fig5_network_4lang.png")

# ============================================================
# Fig 6: LLM top-8 sector bar chart (Table 2b)
# ============================================================
print("=== Fig 6: LLM top sectors ===")
llm_ind = pd.read_csv(BASE / "figures" / "llm_full" / ".." / ".." / "figures" / "causal_network_v5_llm_fdr" / "indicators_llm_all.csv") \
    if (BASE / "figures" / "causal_network_v5_llm_fdr" / "indicators_llm_all.csv").exists() else None
if llm_ind is None:
    llm_ind = pd.read_csv(BASE / "figures" / "causal_network_v5_llm_fdr" / "indicators_llm_all.csv")
pooled = llm_ind.groupby("V1")[["V_total", "H_neg_hostility"]].sum()
pooled["V_share"] = pooled["V_total"] / pooled["V_total"].sum()
pooled["H_share"] = pooled["H_neg_hostility"] / pooled["H_neg_hostility"].sum()
top8 = pooled.sort_values("V_share", ascending=False).head(8)
labels = [short_label(v1, 34) for v1 in top8.index]

fig, ax = plt.subplots(figsize=(10.5, 6))
y = np.arange(len(top8))[::-1]
ax.barh(y + 0.18, top8["V_share"], height=0.32, color=ty.MATLAB_COLORS[0], label="Total (vertical exposure)")
ax.barh(y - 0.18, top8["H_share"], height=0.32, color=ty.MATLAB_COLORS[1], label="Negative-hostility-weighted")
ax.set_yticks(y); ax.set_yticklabels([f"V1={v1}  {l}" for v1, l in zip(top8.index, labels)], fontsize=10)
ax.set_xlabel("Share of pooled corpus exposure")
ax.set_title("Top sectors by aggregate mapped exposure — semantic (LLM) classifier",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGOUT / "fig6_llm_top_sectors.png", dpi=200)
plt.close(fig)
print("  saved fig6_llm_top_sectors.png")

# ============================================================
# B1: PC1 4-stage comparison
# ============================================================
print("=== B1: PC1 comparison ===")
tfidf_raw = {"arabic": 35.6, "chinese": 39.6, "english": 88.5, "persian": 35.0}  # from earlier diagnostic
tfidf_share = pd.read_csv(V4_DIR / "summary_v4_share_fdr.csv").set_index("language")["PC1_pct_corrected"]
llm_raw = pd.read_csv(BASE / "figures" / "causal_network_v5_llm_fdr" / "summary_v5_llm_fdr.csv").set_index("language")["PC1_pct"]
llm_cf = pd.read_csv(BASE / "figures" / "causal_network_v5c_llm_cfadj" / "summary_v5c_cfadj.csv").set_index("language")["PC1_after"]

stages = ["Raw volume\n(TF-IDF)", "Share-transformed\n(TF-IDF)", "Semantic (LLM)\nclassifier", "LLM +\ncommon-factor removed"]
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(LANGS))
width = 0.2
vals = [
    [tfidf_raw[l] for l in LANGS],
    [tfidf_share[l] for l in LANGS],
    [llm_raw[l] for l in LANGS],
    [llm_cf[l] for l in LANGS],
]
colors = [ty.MATLAB_COLORS[i] for i in range(4)]
for i, (v, s, c) in enumerate(zip(vals, stages, colors)):
    ax.bar(x + (i - 1.5) * width, v, width, label=s, color=c)
ax.set_xticks(x); ax.set_xticklabels([LANG_LABEL[l] for l in LANGS], fontsize=12)
ax.set_ylabel("First principal component (% of variance)")
ax.set_title("Common-factor share across classifier and transform choices", fontsize=13, fontweight="bold")
ax.axhline(20, color="0.4", linestyle="--", linewidth=1, alpha=0.7)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGOUT / "figB1_pc1_comparison.png", dpi=200)
plt.close(fig)
print("  saved figB1_pc1_comparison.png")

# ============================================================
# B2: TF-IDF vs LLM sector share dumbbell chart
# ============================================================
print("=== B2: rank change dumbbell ===")
PROB = [f"IO_V1_{i}" for i in range(1, 51)]
TFIDF_FILES = {
    "arabic": "iran_ar_v3_with_io_matrix.csv", "chinese": "iran_ch_v3_with_io_matrix.csv",
    "english": "iran_en_v3_with_io_matrix.csv", "persian": "iran_pe_v3_with_io_matrix.csv",
}
V_tfidf_pool = np.zeros(50)
for lang, fname in TFIDF_FILES.items():
    dft = pd.read_csv(BASE / fname, usecols=PROB, low_memory=False)
    V_tfidf_pool += dft.to_numpy(dtype=float).sum(axis=0)
tfidf_share_all = V_tfidf_pool / V_tfidf_pool.sum()  # index 0..49 -> V1 1..50
llm_share_all = (pooled["V_total"] / pooled["V_total"].sum())  # indexed by V1

top8_tfidf = {45, 26, 44, 12, 9, 47, 28, 43}
top8_llm = {45, 5, 26, 33, 41, 35, 25, 28}
all_sectors = sorted(top8_tfidf | top8_llm)
tf_vals = [tfidf_share_all[v - 1] for v in all_sectors]
llm_vals = [llm_share_all.get(v, 0.0) for v in all_sectors]
order = np.argsort(llm_vals)[::-1]
all_sectors = [all_sectors[i] for i in order]
tf_vals = [tf_vals[i] for i in order]
llm_vals = [llm_vals[i] for i in order]
labels2 = [f"V1={v}  {short_label(v, 26)}" for v in all_sectors]

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(all_sectors))[::-1]
for yi, tf, ll in zip(y, tf_vals, llm_vals):
    ax.plot([tf, ll], [yi, yi], color="0.75", zorder=1, linewidth=2)
ax.scatter(tf_vals, y, color=ty.MATLAB_COLORS[0], s=90, zorder=2, label="TF-IDF (Table 2)")
ax.scatter(llm_vals, y, color=ty.MATLAB_COLORS[1], s=90, zorder=2, label="Semantic / LLM (Table 2b)")
ax.set_yticks(y); ax.set_yticklabels(labels2, fontsize=9.5)
ax.set_xlabel("Aggregate exposure share (total, vertical)")
ax.set_title("Sector composition shifts between classifiers", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGOUT / "figB2_rank_change.png", dpi=200)
plt.close(fig)
print("  saved figB2_rank_change.png")

# ============================================================
# B3: IO alignment — FDR edge rate, IO-adjacent vs non-adjacent
# ============================================================
print("=== B3: IO alignment bar chart ===")
A = pd.read_csv(BASE / "icio_A_50x50.csv", index_col=0).values.astype(float)
q75 = np.quantile(A[A > 0], 0.75)
adj = (A >= q75).astype(float)
mask = ~np.eye(50, dtype=bool)
n_adj_pairs = int(adj[mask].sum())
n_nonadj_pairs = int((1 - adj)[mask].sum())

rates_adj, rates_non = [], []
for lang in LANGS:
    edges_all = pd.read_csv(V4_DIR / f"ty_edges_allpairs_{lang}.csv")
    sig = edges_all[edges_all["fdr_significant"]]
    N = np.zeros((50, 50))
    for r in sig.itertuples(index=False):
        N[int(r.source) - 1, int(r.target) - 1] = 1.0
    both = int(((N == 1) & (adj == 1))[mask].sum())
    non = int(((N == 1) & (adj == 0))[mask].sum())
    rates_adj.append(both / n_adj_pairs * 100)
    rates_non.append(non / n_nonadj_pairs * 100)

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(LANGS))
width = 0.32
ax.bar(x - width/2, rates_adj, width, label="IO-adjacent pairs (top quartile of $a_{ij}$)", color=ty.MATLAB_COLORS[0])
ax.bar(x + width/2, rates_non, width, label="Non-adjacent pairs", color=ty.MATLAB_COLORS[3])
ax.set_xticks(x); ax.set_xticklabels([LANG_LABEL[l] for l in LANGS], fontsize=12)
ax.set_ylabel("Share of pairs selected as FDR-significant edges (%)")
ax.set_title("News lead-lag edges concentrate on input-output-adjacent sector pairs",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGOUT / "figB3_io_alignment.png", dpi=200)
plt.close(fig)
print("  saved figB3_io_alignment.png")

print("\nALL DONE ->", FIGOUT)

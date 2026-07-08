"""Create V1-sector Toda-Yamamoto causal network maps.

The script reads the four article-level IO probability matrix files, converts
IO_V1_* probabilities into daily soft occurrence series, runs pairwise
Toda-Yamamoto lag-augmented causality tests, and saves one MATLAB-style
directed network figure per language.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
VERSION_NAME = "v3"
OUTPUT_DIR = FIG_DIR / f"causal network_{VERSION_NAME}"
LOOKUP_FILE = BASE_DIR / "icio_sectors_with_isic_rev4_descriptions.csv"
PROB_COLS = [f"IO_V1_{i}" for i in range(1, 51)]
LANG_FILES = {
    "Arabic": BASE_DIR / "iran_ar_v3_with_io_matrix.csv",
    "Chinese": BASE_DIR / "iran_ch_v3_with_io_matrix.csv",
    "English": BASE_DIR / "iran_en_v3_with_io_matrix.csv",
    "Persian": BASE_DIR / "iran_pe_v3_with_io_matrix.csv",
}

ALPHA = 0.05
MAX_LAG = 7
MAX_EDGES = 10000
MIN_EDGE_ABS_R = 0.7

MATLAB_COLORS = np.array(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ]
)


@dataclass(frozen=True)
class TYResult:
    source: int
    target: int
    p_lag: int
    dmax: int
    lead_days: int
    ty_pvalue: float
    pearson_r: float
    abs_r: float


def set_matlab_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "font.family": "Arial",
            "font.size": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(LOOKUP_FILE)
    lookup = lookup.rename(columns={"ISIC Rev.4": "ISIC_Rev4"})
    lookup["V1"] = lookup["V1"].astype(int)
    return lookup


def read_daily_occurrence(path: Path) -> pd.DataFrame:
    usecols = ["published_at"] + PROB_COLS
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["date"] = pd.to_datetime(
        df["published_at"], errors="coerce", dayfirst=True
    ).dt.floor("D")
    df = df.dropna(subset=["date"])
    for col in PROB_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    daily = df.groupby("date", sort=True)[PROB_COLS].sum()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily.columns = [int(col.rsplit("_", 1)[1]) for col in daily.columns]
    return daily


def select_nodes(daily: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    total = daily.sum(axis=0)
    active_days = daily.gt(0).sum(axis=0)
    nodes = pd.DataFrame(
        {
            "V1": total.index.astype(int),
            "total_occurrence": total.to_numpy(dtype=float),
            "active_days": active_days.to_numpy(dtype=int),
        }
    )
    nodes = nodes.sort_values("V1")
    nodes = nodes.merge(lookup, on="V1", how="left")
    return nodes


def integration_order(series: pd.Series) -> int:
    y = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if y.nunique() <= 1:
        return 0
    for d in range(3):
        test_y = y.diff(d).dropna() if d else y.dropna()
        if len(test_y) < 12 or test_y.nunique() <= 1:
            return d
        try:
            pvalue = adfuller(test_y, autolag="AIC")[1]
        except Exception:
            return d
        if pvalue < 0.05:
            return d
    return 2


def choose_var_lag(pair: pd.DataFrame, max_lag: int) -> int:
    maxlags = min(max_lag, max(1, (len(pair) - 8) // 4))
    if maxlags <= 1:
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = VAR(pair).select_order(maxlags=maxlags)
        lag = selected.selected_orders.get("aic")
        if lag is None or lag < 1 or not np.isfinite(lag):
            lag = selected.selected_orders.get("bic")
        if lag is None or lag < 1 or not np.isfinite(lag):
            return 1
        return int(min(max_lag, max(1, lag)))
    except Exception:
        return 1


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return 0.0
    r = a.corr(b)
    return 0.0 if pd.isna(r) else float(r)


def best_positive_lead_corr(source: pd.Series, target: pd.Series) -> tuple[int, float]:
    """Return lag where source best leads target over 1..MAX_LAG days."""
    best_lag = 1
    best_r = 0.0
    for lag in range(1, MAX_LAG + 1):
        a = source.iloc[:-lag].reset_index(drop=True)
        b = target.iloc[lag:].reset_index(drop=True)
        r = pearson_corr(a, b)
        if abs(r) > abs(best_r):
            best_lag = lag
            best_r = r
    return best_lag, best_r


def ty_wald_pvalue(
    data: pd.DataFrame, source_col: str, target_col: str, p_lag: int, dmax: int
) -> float:
    total_lag = p_lag + dmax
    rows = []
    y = []
    for t in range(total_lag, len(data)):
        row = {}
        for lag in range(1, total_lag + 1):
            row[f"{target_col}_L{lag}"] = data[target_col].iloc[t - lag]
            row[f"{source_col}_L{lag}"] = data[source_col].iloc[t - lag]
        rows.append(row)
        y.append(data[target_col].iloc[t])
    if len(rows) <= 2 * total_lag + 2:
        return math.nan

    x = sm.add_constant(pd.DataFrame(rows), has_constant="add")
    y_arr = np.asarray(y, dtype=float)
    fit = sm.OLS(y_arr, x).fit()

    test_cols = [f"{source_col}_L{lag}" for lag in range(1, p_lag + 1)]
    param_names = list(fit.params.index)
    r_matrix = np.zeros((len(test_cols), len(param_names)))
    for row_idx, name in enumerate(test_cols):
        if name not in param_names:
            return math.nan
        r_matrix[row_idx, param_names.index(name)] = 1.0

    try:
        test = fit.wald_test(r_matrix, scalar=True)
        pvalue = float(test.pvalue)
        if np.isfinite(pvalue):
            return pvalue
    except Exception:
        pass

    # Fallback for singular covariance cases: Wald statistic with pseudo-inverse.
    beta = fit.params.to_numpy()
    cov = fit.cov_params().to_numpy()
    rb = r_matrix @ beta
    rcov = r_matrix @ cov @ r_matrix.T
    try:
        stat = float(rb.T @ np.linalg.pinv(rcov) @ rb)
    except Exception:
        return math.nan
    return float(chi2.sf(stat, len(test_cols)))


def run_ty_tests(daily: pd.DataFrame, v1_nodes: list[int]) -> pd.DataFrame:
    orders = {v1: integration_order(daily[v1]) for v1 in v1_nodes}
    results: list[TYResult] = []
    for i, source in enumerate(v1_nodes):
        for target in v1_nodes:
            if source == target:
                continue
            pair = daily[[source, target]].copy()
            pair.columns = ["source", "target"]
            if pair["source"].nunique() <= 1 or pair["target"].nunique() <= 1:
                continue
            p_lag = choose_var_lag(pair, MAX_LAG)
            dmax = max(orders[source], orders[target])
            pvalue = ty_wald_pvalue(pair, "source", "target", p_lag, dmax)
            if not np.isfinite(pvalue):
                continue
            lead_days, r = best_positive_lead_corr(daily[source], daily[target])
            results.append(
                TYResult(
                    source=source,
                    target=target,
                    p_lag=p_lag,
                    dmax=dmax,
                    lead_days=lead_days,
                    ty_pvalue=pvalue,
                    pearson_r=r,
                    abs_r=abs(r),
                )
            )
    edges = pd.DataFrame([r.__dict__ for r in results])
    if edges.empty:
        return edges
    edges = edges[edges["ty_pvalue"].lt(ALPHA)].copy()
    edges = edges.sort_values(["ty_pvalue", "abs_r"], ascending=[True, False])
    if len(edges) > MAX_EDGES:
        edges = edges.head(MAX_EDGES)
    return edges.reset_index(drop=True)


def compute_lead_scores(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[int, float]:
    scores = {int(v1): 0.0 for v1 in nodes["V1"]}
    if edges.empty:
        return scores
    for row in edges.itertuples(index=False):
        weight = max(float(row.abs_r), 0.05)
        scores[int(row.source)] += weight
        scores[int(row.target)] -= weight
    return scores


def network_positions(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[int, tuple[float, float]]:
    ordered = list(range(1, 51))
    angles = [np.pi / 2 - 2 * np.pi * (v1 - 1) / 50 for v1 in ordered]
    radius_x = 0.43
    radius_y = 0.39
    return {
        v1: (0.5 + radius_x * math.cos(angle), 0.50 + radius_y * math.sin(angle))
        for v1, angle in zip(ordered, angles)
    }


def save_network_figure(
    language: str, nodes: pd.DataFrame, edges: pd.DataFrame, output_path: Path
) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["V1"].astype(int).tolist())
    for row in edges.itertuples(index=False):
        graph.add_edge(
            int(row.source),
            int(row.target),
            ty_pvalue=float(row.ty_pvalue),
            pearson_r=float(row.pearson_r),
            abs_r=float(row.abs_r),
        )

    pos = network_positions(nodes, edges)
    counts = nodes.set_index("V1")["total_occurrence"].to_dict()
    max_count = max(counts.values()) if counts else 1.0
    node_sizes = {
        int(v1): 120 + 520 * math.sqrt(float(counts[int(v1)]) / max_count)
        for v1 in graph.nodes
    }

    fig, ax = plt.subplots(figsize=(13.5, 13.5))
    ax.set_title(
        f"Toda-Yamamoto Causal Network Map: {language} V1 Sectors",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if edges.empty:
        ax.text(
            0.5,
            0.5,
            "No TY-significant directed edges",
            ha="center",
            va="center",
            fontsize=16,
            color="0.35",
        )

    edge_widths = [
        1.0 + 5.4 * graph[u][v]["abs_r"] for u, v in graph.edges()
    ]
    edge_colors = [
        MATLAB_COLORS[0]
        for u, v in graph.edges()
    ]
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.44,
        connectionstyle="arc3,rad=0.10",
        min_source_margin=7,
        min_target_margin=9,
    )

    nodelist = list(graph.nodes)
    node_colors = [
        MATLAB_COLORS[i % len(MATLAB_COLORS)] for i, _ in enumerate(nodelist)
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodelist,
        node_size=[node_sizes[v1] for v1 in nodelist],
        node_color=node_colors,
        edgecolors="white",
        linewidths=1.5,
        alpha=0.96,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={v1: str(v1) for v1 in graph.nodes},
        font_color="white",
        font_size=7,
        font_weight="bold",
        ax=ax,
    )

    edge_labels = {
        (int(row.source), int(row.target)): f"+{int(row.lead_days)}d"
        for row in edges.itertuples(index=False)
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=5.2,
        font_color=MATLAB_COLORS[0],
        rotate=False,
        label_pos=0.52,
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.58},
        ax=ax,
    )

    subtitle = (
        f"50 V1 sectors | {len(edges)} TY-significant directed edges "
        f"(alpha={ALPHA}, |r|>={MIN_EDGE_ABS_R}, max lag={MAX_LAG}d)"
    )
    ax.text(
        0.5,
        0.965,
        subtitle,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    note = (
        "Node label = V1 number only. Node size proportional to sqrt(total daily soft occurrence). "
        "Arrow direction from Toda-Yamamoto Wald test; edge thickness proportional to |Pearson r|. "
        "Edge label shows source lead days; full TY p-values and correlations are in CSV."
    )
    fig.text(0.02, 0.02, "Note. " + note, ha="left", va="bottom", fontsize=9, color="0.25")

    legend_handles = [
        Line2D([0], [0], color=MATLAB_COLORS[0], lw=4, label="TY-significant directed edge"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.55",
            linestyle="None",
            markersize=10,
            label="Node size = occurrence",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=True,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.12, top=0.88)
    fig.savefig(output_path)
    plt.close(fig)


def slug(language: str) -> str:
    return language.lower()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_matlab_style()
    lookup = read_lookup()
    all_edges = []
    all_nodes = []

    for language, path in LANG_FILES.items():
        daily = read_daily_occurrence(path)
        nodes = select_nodes(daily, lookup)
        v1_nodes = nodes["V1"].astype(int).tolist()
        edges = run_ty_tests(daily, v1_nodes)
        if not edges.empty:
            edges = edges[edges["abs_r"].ge(MIN_EDGE_ABS_R)].reset_index(drop=True)

        nodes = nodes.copy()
        nodes.insert(0, "language", language)
        edges = edges.copy()
        if not edges.empty:
            edges.insert(0, "language", language)

        lang_slug = slug(language)
        daily[v1_nodes].to_csv(
            OUTPUT_DIR / f"ty_v1_daily_occurrence_{lang_slug}_{VERSION_NAME}.csv"
        )
        nodes.to_csv(
            OUTPUT_DIR / f"ty_v1_network_nodes_{lang_slug}_{VERSION_NAME}.csv",
            index=False,
        )
        edges.to_csv(
            OUTPUT_DIR / f"ty_v1_network_edges_{lang_slug}_{VERSION_NAME}.csv",
            index=False,
        )
        save_network_figure(
            language,
            nodes,
            edges,
            OUTPUT_DIR / f"fig08_ty_v1_network_{lang_slug}_{VERSION_NAME}.png",
        )
        all_nodes.append(nodes)
        all_edges.append(edges)
        print(
            f"{language}: saved {len(nodes)} nodes, {len(edges)} edges "
            f"to fig08_ty_v1_network_{lang_slug}_{VERSION_NAME}.png"
        )

    pd.concat(all_nodes, ignore_index=True).to_csv(
        OUTPUT_DIR / f"ty_v1_network_nodes_all_{VERSION_NAME}.csv", index=False
    )
    if all_edges:
        pd.concat(all_edges, ignore_index=True).to_csv(
            OUTPUT_DIR / f"ty_v1_network_edges_all_{VERSION_NAME}.csv", index=False
        )


if __name__ == "__main__":
    main()

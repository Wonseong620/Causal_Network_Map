"""Create MATLAB-style figures from article-level IO probability matrices.

Inputs are the four *_with_io_matrix.csv files. Each row contains an article
and 50 probability columns, IO_V1_1 through IO_V1_50. This script computes
language, publisher-location, risk-type, weekly, entropy, and aggregate V1
metrics, then writes publication-style PNG figures and supporting CSV tables.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
LOOKUP_FILE = BASE_DIR / "icio_sectors_with_isic_rev4_descriptions.csv"
PROB_COLS = [f"IO_V1_{i}" for i in range(1, 51)]
BASE_COLS = [
    "published_at",
    "publisher_location",
    "negativity",
    "war_related",
    "type_risk",
    "economic_subtopic",
]

LANG_FILES = {
    "Arabic": BASE_DIR / "iran_ar_v3_with_io_matrix.csv",
    "Chinese": BASE_DIR / "iran_ch_v3_with_io_matrix.csv",
    "English": BASE_DIR / "iran_en_v3_with_io_matrix.csv",
    "Persian": BASE_DIR / "iran_pe_v3_with_io_matrix.csv",
}

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

RISK_ORDER = [
    "Military",
    "Economic",
    "Maritime",
    "Diplomatic",
    "Nuclear",
    "Proxy",
    "Humanitarian",
    "Cyber",
]


def add_note(fig, text: str, y: float = 0.01) -> None:
    """Add a compact explanatory note to a figure."""
    fig.text(
        0.01,
        y,
        "Note. " + text,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.25",
        wrap=True,
    )


def set_matlab_style() -> None:
    """Apply a clean white MATLAB-like plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_lookup() -> pd.DataFrame:
    """Read the 50-sector ICIO lookup table."""
    lookup = pd.read_csv(LOOKUP_FILE, dtype=str, encoding="utf-8-sig")
    lookup["V1"] = lookup["V1"].astype(int)
    lookup = lookup.sort_values("V1").reset_index(drop=True)
    lookup["short_label"] = lookup["V1"].astype(str) + " " + lookup["Code"]
    return lookup


def read_language_file(path: Path) -> pd.DataFrame:
    """Read one language matrix file and coerce analysis variables."""
    usecols = lambda c: c in BASE_COLS or c in PROB_COLS
    df = pd.read_csv(path, usecols=usecols, dtype=str, encoding="utf-8-sig")
    for col in PROB_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["negativity"] = pd.to_numeric(df.get("negativity"), errors="coerce").fillna(0.0)
    df["war_related"] = pd.to_numeric(df.get("war_related"), errors="coerce").fillna(0.0)
    df["type_risk"] = df.get("type_risk", "").fillna("").replace("", "Unspecified")
    df["publisher_location"] = (
        df.get("publisher_location", "").fillna("").replace("", "Unspecified")
    )
    df["economic_subtopic"] = (
        df.get("economic_subtopic", "").fillna("").replace("", "Unspecified")
    )
    df["date"] = pd.to_datetime(df.get("published_at"), errors="coerce", dayfirst=True)
    return df


def compute_metrics(data: dict[str, pd.DataFrame], lookup: pd.DataFrame) -> dict:
    """Compute reusable tabular metrics for all figures."""
    lang_sector_rows = []
    pub_location_rows = []
    risk_rows = []
    weekly_rows = []
    entropy_rows = []

    for lang, df in data.items():
        probs = df[PROB_COLS].to_numpy(dtype=float)
        negativity = df["negativity"].to_numpy(dtype=float)
        war = df["war_related"].to_numpy(dtype=float)
        shock_weight = negativity * war
        exposure = probs.sum(axis=0)
        shock = (probs * shock_weight[:, None]).sum(axis=0)
        weighted_neg = np.divide(
            (probs * negativity[:, None]).sum(axis=0),
            exposure,
            out=np.zeros_like(exposure),
            where=exposure > 0,
        )

        for idx, row in lookup.iterrows():
            sector_idx = idx
            lang_sector_rows.append(
                {
                    "language": lang,
                    "V1": int(row["V1"]),
                    "Code": row["Code"],
                    "label": row["short_label"],
                    "Industry": row["Industry"],
                    "exposure": exposure[sector_idx],
                    "shock": shock[sector_idx],
                    "weighted_negativity": weighted_neg[sector_idx],
                }
            )

        for location in sorted(df["publisher_location"].unique()):
            if location == "Unspecified":
                continue
            mask = df["publisher_location"].eq(location).to_numpy()
            if not mask.any():
                continue
            loc_probs = probs[mask]
            loc_weight = shock_weight[mask]
            loc_shock = (loc_probs * loc_weight[:, None]).sum(axis=0)
            loc_articles = int(mask.sum())
            for idx, row in lookup.iterrows():
                pub_location_rows.append(
                    {
                        "language": lang,
                        "publisher_location": location,
                        "article_count": loc_articles,
                        "V1": int(row["V1"]),
                        "Code": row["Code"],
                        "label": row["short_label"],
                        "Industry": row["Industry"],
                        "shock": loc_shock[idx],
                    }
                )

        for risk in RISK_ORDER:
            mask = df["type_risk"].eq(risk).to_numpy()
            if mask.any():
                risk_exposure = probs[mask].sum(axis=0)
            else:
                risk_exposure = np.zeros(50)
            for idx, row in lookup.iterrows():
                risk_rows.append(
                    {
                        "language": lang,
                        "type_risk": risk,
                        "V1": int(row["V1"]),
                        "label": row["short_label"],
                        "risk_exposure": risk_exposure[idx],
                    }
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy = -(probs * log_probs).sum(axis=1) / math.log(50)
        entropy_rows.extend(
            {"language": lang, "entropy": val} for val in entropy if np.isfinite(val)
        )

        valid_date = df["date"].notna()
        if valid_date.any():
            tmp = df.loc[valid_date, ["date", "negativity", "war_related"]].copy()
            tmp["week"] = tmp["date"].dt.to_period("W").dt.start_time
            tmp_probs = df.loc[valid_date, PROB_COLS].to_numpy(dtype=float)
            tmp_weight = (
                tmp["negativity"].to_numpy(dtype=float)
                * tmp["war_related"].to_numpy(dtype=float)
            )
            for sector_pos, row in lookup.iterrows():
                weighted = tmp_probs[:, sector_pos] * tmp_weight
                weekly = (
                    pd.DataFrame({"week": tmp["week"].to_numpy(), "shock": weighted})
                    .groupby("week", as_index=False)["shock"]
                    .sum()
                )
                for _, wrow in weekly.iterrows():
                    weekly_rows.append(
                        {
                            "language": lang,
                            "week": wrow["week"],
                            "V1": int(row["V1"]),
                            "label": row["short_label"],
                            "shock": wrow["shock"],
                        }
                    )

    return {
        "sector": pd.DataFrame(lang_sector_rows),
        "publisher_location": pd.DataFrame(pub_location_rows),
        "risk": pd.DataFrame(risk_rows),
        "weekly": pd.DataFrame(weekly_rows),
        "entropy": pd.DataFrame(entropy_rows),
    }


def save_language_sector_heatmap(metrics: dict, lookup: pd.DataFrame) -> None:
    sector = metrics["sector"]
    top_v1 = (
        sector.groupby("V1")["shock"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    pivot = (
        sector[sector["V1"].isin(top_v1)]
        .pivot(index="language", columns="V1", values="shock")
        .reindex(index=LANG_FILES.keys(), columns=top_v1)
    )
    row_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5.4))
    im = ax.imshow(row_norm.to_numpy(), aspect="auto", cmap="parula" if "parula" in plt.colormaps() else "viridis")
    ax.set_title("Language-Sector Integrated Shock Share")
    ax.set_xlabel("ICIO sector, V1 number")
    ax.set_ylabel("Language")
    ax.set_xticks(range(len(top_v1)))
    ax.set_xticklabels(top_v1)
    ax.set_yticks(range(len(row_norm.index)))
    ax.set_yticklabels(row_norm.index)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-language shock share")
    add_note(
        fig,
        "Shock_s,l = sum_i(IO_V1_s,i x negativity_i x war_related_i), normalized within language. "
        "Darker cells identify which sectors carry each language's war-narrative economic exposure.",
    )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG_DIR / "fig01_language_sector_integrated_shock_heatmap.png")
    plt.close(fig)


def save_sector_rank_comparison(metrics: dict) -> None:
    sector = metrics["sector"]
    top_v1 = (
        sector.groupby("V1")["shock"]
        .sum()
        .sort_values(ascending=False)
        .head(18)
        .index.tolist()
    )
    sub = sector[sector["V1"].isin(top_v1)].copy()
    sub["within_lang_share"] = sub["shock"] / sub.groupby("language")["shock"].transform("sum")

    fig, ax = plt.subplots(figsize=(14, 6.6))
    offsets = np.linspace(-0.27, 0.27, len(LANG_FILES))
    for i, lang in enumerate(LANG_FILES.keys()):
        lang_sub = sub[sub["language"].eq(lang)].sort_values("V1")
        ax.scatter(
            lang_sub["V1"] + offsets[i],
            lang_sub["within_lang_share"],
            color=MATLAB_COLORS[i],
            s=55,
            edgecolor="black",
            linewidth=0.4,
            label=lang,
            zorder=3,
        )
    ax.set_title("Cross-Language Sector Shock Concentration")
    ax.set_xlabel("ICIO sector, V1 number")
    ax.set_ylabel("Share of language-specific integrated shock")
    ax.set_xticks(sorted(top_v1))
    ax.set_ylim(bottom=0)
    ax.legend(ncol=4, loc="upper right", frameon=True)
    add_note(
        fig,
        "Each marker is a sector's share of total integrated shock within a language. "
        "This compares narrative concentration across languages and highlights sectors emphasized differently by media language.",
    )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG_DIR / "fig02_cross_language_sector_shock_rank.png")
    plt.close(fig)


def save_publisher_location_shock(metrics: dict) -> None:
    pub = metrics["publisher_location"].copy()
    if pub.empty:
        return

    total_by_location = (
        pub.groupby("publisher_location")
        .agg(shock=("shock", "sum"), article_count=("article_count", "max"))
        .query("article_count >= 50")
        .sort_values("shock", ascending=False)
        .head(14)
    )
    top_locations = total_by_location.index.tolist()
    top_v1 = (
        pub[pub["publisher_location"].isin(top_locations)]
        .groupby("V1")["shock"]
        .sum()
        .sort_values(ascending=False)
        .head(18)
        .index.tolist()
    )
    sub = pub[pub["publisher_location"].isin(top_locations) & pub["V1"].isin(top_v1)]
    pivot = (
        sub.pivot_table(
            index="publisher_location",
            columns="V1",
            values="shock",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=top_locations, columns=top_v1)
    )
    row_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7.2))
    im = ax.imshow(
        row_norm.to_numpy(),
        aspect="auto",
        cmap="parula" if "parula" in plt.colormaps() else "viridis",
    )
    ax.set_title("Publisher-Location Sector Shock Concentration")
    ax.set_xlabel("ICIO sector, V1 number")
    ax.set_ylabel("Publisher location")
    ax.set_xticks(range(len(top_v1)))
    ax.set_xticklabels(top_v1)
    ax.set_yticks(range(len(top_locations)))
    ylabels = [
        f"{loc} (n={int(total_by_location.loc[loc, 'article_count'])})"
        for loc in top_locations
    ]
    ax.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-location shock share")
    add_note(
        fig,
        "Rows are publisher_location values with at least 50 articles, excluding blank locations. "
        "Shock_s = sum(IO_V1_s x negativity x war_related), normalized within publisher location. "
        "The figure shows how geographic media locations concentrate war-narrative economic exposure in different IO sectors.",
    )
    fig.subplots_adjust(left=0.20, right=0.93, bottom=0.16)
    fig.savefig(FIG_DIR / "fig06_cross_publocation_sector_shock.png")
    plt.close(fig)


def save_risk_heatmaps(metrics: dict) -> None:
    sector = metrics["sector"]
    risk = metrics["risk"]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    axes = axes.ravel()
    for ax, lang in zip(axes, LANG_FILES.keys()):
        top_v1 = (
            sector[sector["language"].eq(lang)]
            .sort_values("shock", ascending=False)
            .head(10)["V1"]
            .tolist()
        )
        sub = risk[risk["language"].eq(lang) & risk["V1"].isin(top_v1)]
        pivot = (
            sub.pivot(index="type_risk", columns="V1", values="risk_exposure")
            .reindex(index=RISK_ORDER, columns=top_v1)
            .fillna(0)
        )
        row_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        im = ax.imshow(row_norm.to_numpy(), aspect="auto", cmap="hot")
        ax.set_title(lang)
        ax.set_xticks(range(len(top_v1)))
        ax.set_xticklabels(top_v1)
        ax.set_yticks(range(len(RISK_ORDER)))
        ax.set_yticklabels(RISK_ORDER)
    fig.suptitle("Risk Type Composition Across Top IO Sectors", y=0.99)
    cax = fig.add_axes([0.93, 0.20, 0.015, 0.62])
    fig.colorbar(im, cax=cax, label="Risk-type sector share")
    add_note(
        fig,
        "Rows are article type_risk categories; columns are top V1 sectors within each language. "
        "Cell values are risk-type-normalized IO exposure shares, showing whether sectors are linked to military, economic, maritime, or diplomatic narratives.",
        y=0.005,
    )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.15, top=0.92, wspace=0.34, hspace=0.42)
    fig.savefig(FIG_DIR / "fig03_sector_risk_type_heatmaps.png")
    plt.close(fig)


def save_weekly_trends(metrics: dict) -> None:
    sector = metrics["sector"]
    weekly = metrics["weekly"]
    legend_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.ravel()
    for ax, lang in zip(axes, LANG_FILES.keys()):
        top_rows = (
            sector[sector["language"].eq(lang)]
            .sort_values("shock", ascending=False)
            .head(5)[["V1", "Industry"]]
        )
        top_v1 = top_rows["V1"].tolist()
        legend_labels = {
            row["V1"]: f"V1 {int(row['V1'])}"
            for _, row in top_rows.iterrows()
        }
        for _, row in top_rows.iterrows():
            legend_rows.append(
                {
                    "language": lang,
                    "V1": int(row["V1"]),
                    "Industry": row["Industry"],
                    "short_legend": f"V1 {int(row['V1'])}",
                }
            )
        sub = weekly[weekly["language"].eq(lang) & weekly["V1"].isin(top_v1)]
        for i, v1 in enumerate(top_v1):
            line = sub[sub["V1"].eq(v1)].sort_values("week")
            ax.plot(
                line["week"],
                line["shock"],
                color=MATLAB_COLORS[i % len(MATLAB_COLORS)],
                linewidth=1.6,
                label=legend_labels[v1],
            )
        ax.set_title(lang)
        ax.set_ylabel("Weekly integrated shock")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper right", frameon=True, fontsize=7, ncol=1)
    fig.suptitle("Weekly Sector Shock Trends, Top 5 Sectors by Language", y=0.98)
    add_note(
        fig,
        "Weekly shock sums IO sector probability weighted by negativity and war_related. "
        "The trend traces when language-specific narratives intensify around their top V1 sectors. "
        "Legend uses V1 numbers to avoid overlap; see fig04_weekly_top_sector_legend.csv for V1-to-Industry names.",
        y=0.005,
    )
    fig.subplots_adjust(bottom=0.15, hspace=0.38, wspace=0.22, right=0.96)
    fig.savefig(FIG_DIR / "fig04_weekly_top_sector_shock_trends.png")
    plt.close(fig)
    pd.DataFrame(legend_rows).to_csv(
        FIG_DIR / "fig04_weekly_top_sector_legend.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_entropy_distribution(metrics: dict) -> None:
    entropy = metrics["entropy"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 1, 31)
    for i, lang in enumerate(LANG_FILES.keys()):
        vals = entropy.loc[entropy["language"].eq(lang), "entropy"].to_numpy()
        ax.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=MATLAB_COLORS[i],
            label=lang,
        )
    ax.set_title("Article-Level IO Distribution Entropy")
    ax.set_xlabel("Normalized entropy, 0=single-sector and 1=uniform")
    ax.set_ylabel("Article count")
    ax.legend(frameon=True)
    add_note(
        fig,
        "Entropy is computed from IO_V1_1...IO_V1_50 and normalized by log(50). "
        "Low entropy indicates a concentrated sector shock; high entropy indicates diffuse or low-information cross-sector exposure.",
    )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG_DIR / "fig05_io_entropy_distribution.png")
    plt.close(fig)


def save_v1_aggregate_stacked_bar(metrics: dict) -> None:
    sector = metrics["sector"]
    pivot = (
        sector.pivot(index="V1", columns="language", values="exposure")
        .reindex(index=range(1, 51), columns=LANG_FILES.keys())
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(15, 6.8))
    bottom = np.zeros(len(pivot))
    x = pivot.index.to_numpy()
    for i, lang in enumerate(LANG_FILES.keys()):
        vals = pivot[lang].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=MATLAB_COLORS[i],
            alpha=0.85,
            width=0.82,
            label=lang,
        )
        bottom += vals
    ax.set_xlabel("ICIO sector, V1 number")
    ax.set_ylabel("Aggregate vertical sum")
    ax.set_title("V1-Level Aggregate Shock Index by Language")
    ax.set_xticks(range(1, 51))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(ncol=4, loc="upper right", frameon=True)
    add_note(
        fig,
        "Stacked bars show matrix vertical sums by language: sum_i(IO_V1_s,i). "
        "Each article contributes total mass 1 across 50 sectors, so the total stacked mass equals the total number of articles. "
        "Color segments show how each language contributes to the aggregate V1 shock index.",
    )
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(FIG_DIR / "fig07_v1_aggregate_shock_index.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for old_png in FIG_DIR.glob("fig*.png"):
        old_png.unlink()
    set_matlab_style()
    lookup = read_lookup()
    data = {lang: read_language_file(path) for lang, path in LANG_FILES.items()}
    metrics = compute_metrics(data, lookup)

    save_language_sector_heatmap(metrics, lookup)
    save_sector_rank_comparison(metrics)
    save_publisher_location_shock(metrics)
    save_risk_heatmaps(metrics)
    save_weekly_trends(metrics)
    save_entropy_distribution(metrics)
    save_v1_aggregate_stacked_bar(metrics)

    metrics["sector"].to_csv(FIG_DIR / "sector_metrics_by_language.csv", index=False)
    metrics["publisher_location"].to_csv(
        FIG_DIR / "sector_metrics_by_publisher_location.csv", index=False
    )
    print(f"Saved figures and sector_metrics_by_language.csv in {FIG_DIR}")


if __name__ == "__main__":
    main()

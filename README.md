# Qualitative Shock Mapping for Quantitative Production Network Models: Evidence from Iran-Related Hostilities

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21257558.svg)](https://doi.org/10.5281/zenodo.21257558)

https://github.com/Wonseong620/Causal_Network_Map

The research question would be: "How can qualitative news about regional hostilities be converted into structured sectoral shock indicators, and how can these indicators supplement quantitative models of production network spillovers?"

In periods of regional hostilities, economic shocks are often observed first through qualitative evidence rather than official statistics. News reports, policy announcements, market commentary, and firm-level disclosures provide early information about disruptions, but this information is rarely organized in a form that can be directly used in quantitative models. This study addresses that gap by developing a news-based sectoral shock-mapping framework. The framework translates qualitative evidence into structured sectoral shock indicators, links them to input-output sectors, and uses dynamic network methods to examine propagation across the production system. The resulting approach serves as a supplement to quantitative modelling by improving the identification, timing, and sectoral allocation of geopolitical shocks.

## Live viewer

https://wonseong620.github.io/Causal_Network_Map/

### Usage

The **Topic** list at the top of the filters is ordered by year (`yyyy_event`). The current dataset covers the 2026 Iran war; the other global events listed (COVID-19, Ukraine war) are placeholders marked TBD until their corpora are processed.

The page has three columns: filters on the left (Data, Edges, Display, Export), the network in the centre, and results on the right (selected sector card, summary statistics, edges by sector group, legends). Below 1240px the results move under the network; below 820px the filters collapse behind a **Filters** button.

- **Data** selects the dataset: **v5** (share-transformed series, BH-FDR edge selection; the manuscript's specification of record) or the legacy **v3** (raw volume series).
- **Language** switches between the Arabic, Chinese, English, and Persian corpora (each ~80KB, loaded on demand).
- **Correlation threshold (|r|)** and **Max lead day** filter edges by the lagged Pearson correlation at the selected lead. Dashed orange edges indicate negative correlations.
- **Edge significance** restricts edges to BH-FDR-significant pairs (v5) or TY-significant pairs (v3), raw p < 0.05, or shows all correlations.
- **Edge display: Hover / selected node** shows only edges incident to the hovered or clicked (locked) sector; the **role** filter isolates edges the sector leads or lags. Nodes are keyboard-accessible (Tab + Enter). Edge labels (`+d`, source lead days) appear when at most 60 edges are visible and always for a focused node.
- The side panel shows a 7 x 7 matrix of visible edges by sector group (row leads column), and a focused node is named beside the ring.
- **Compare all (2 x 2)** renders all four language networks under matched filters — the layout used for the four-language comparison figure in the manuscript. Hovering or clicking a sector in any panel highlights it in all four, keeping only its incident edges (the role filter applies), and the sector card lists its lead/lag counts per language.
- **Export** downloads the current view as PNG (2200 px wide) or SVG.

Node colours mark the seven ICIO V1 sector groups (agriculture & mining 1–8, light & process manufacturing 9–19, machinery & transport equipment 20–27, utilities & construction 28–30, trade, transport & logistics 31–37, information, finance & business services 38–44, public & social services 45–50); the outer arcs trace the same groups. The palette is shared with the manuscript figures and was checked for colour-vision deficiency on adjacent arcs.

URL parameters preset every control, e.g. `?dataset=v5&language=Chinese&edgeMode=hover&node=16&nodeRole=lead`. Adding `export=paper` hides the page chrome, enlarges labels for print, and adds an in-figure legend; `replication/code/export_paper_figures.mjs` uses this mode to print the manuscript's network figures to vector PDF with headless Chromium.

### Data

`data/v5/<language>.json` — 50 ICIO V1 sector nodes and all 2,450 ordered sector pairs per language from the replication package (share-transformed series, BH-FDR 5%). Built by `replication/code/build_viewer_v5.py`. Edge row format: `[source_v1, target_v1, fdr_significant, ty_pvalue, ty_lag, lead_days, pearson_r]`.

`data/v3/<language>.json` — legacy raw-volume series with 7-day lagged correlations, kept for comparison. Edge row format: `[source_v1, target_v1, ty_significant, ty_pvalue, ty_lag, r_lead1 ... r_lead7]`.

## Replication package

A curated replication package for the paper is in [`replication/`](replication/):
derived daily sectoral indicators and within-day share-transformed series,
article-level sector-probability vectors (lexical TF-IDF and zero-shot LLM
classifiers), Toda-Yamamoto lead-lag edge tables with false-discovery-rate
flags, the world-average input-output coefficient matrix and alignment
statistics, the 50-sector ICIO V1 lookup, the LLM classification prompt, and all
analysis/figure scripts. See [`replication/README.md`](replication/README.md) for
the full manifest and a mapping to the paper.

**Raw article text is not redistributed** for copyright reasons; only derived
quantities and non-text metadata are included. The live viewer's default
`data/v5/` dataset is built from these files.

## Citation

If you use this software or data, please cite the accompanying article and this
archive (see [`CITATION.cff`](CITATION.cff)). The archived releases are on
Zenodo — concept DOI (all versions):
[10.5281/zenodo.21257558](https://doi.org/10.5281/zenodo.21257558); this release
(v1.0.0): [10.5281/zenodo.21257559](https://doi.org/10.5281/zenodo.21257559).

## License

- **Code** — the interactive viewer and the scripts under `replication/code/` —
  is released under the [MIT License](LICENSE).
- **Data** under `replication/data/` is released under
  [CC BY 4.0](replication/data/LICENSE).

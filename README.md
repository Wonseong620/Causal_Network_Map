# Qualitative Shock Mapping for Quantitative Production Network Models: Evidence from Iran-Related Hostilities

https://github.com/Wonseong620/Causal_Network_Map

The research question would be: "How can qualitative news about regional hostilities be converted into structured sectoral shock indicators, and how can these indicators supplement quantitative models of production network spillovers?"

In periods of regional hostilities, economic shocks are often observed first through qualitative evidence rather than official statistics. News reports, policy announcements, market commentary, and firm-level disclosures provide early information about disruptions, but this information is rarely organized in a form that can be directly used in quantitative models. This study addresses that gap by developing a news-based sectoral shock-mapping framework. The framework translates qualitative evidence into structured sectoral shock indicators, links them to input-output sectors, and uses dynamic network methods to examine propagation across the production system. The resulting approach serves as a supplement to quantitative modelling by improving the identification, timing, and sectoral allocation of geopolitical shocks.

## Live viewer

https://wonseong620.github.io/Causal_Network_Map/

### Usage

- **Language** switches between the Arabic, Chinese, English, and Persian corpora (each ~170KB, loaded on demand).
- **Correlation threshold (|r|)** and **Max lead day** filter edges by the largest absolute lagged Pearson correlation within the lead window. Dashed orange edges indicate negative correlations.
- **TY test significance** restricts edges to those passing the Toda-Yamamoto Wald test.
- **Edge display: Hover / selected node** shows only edges incident to the hovered or clicked (locked) sector; the **role** filter isolates edges the sector leads or lags. Nodes are keyboard-accessible (Tab + Enter).
- **Compare all (2 x 2)** renders all four language networks under matched filters — the layout used for the four-language comparison figure in the manuscript.
- **Download PNG** exports the current view at 2200 x 2200.

### Data

`data/v3/<language>.json` — 50 ICIO V1 sector nodes and all 2,450 ordered sector pairs per language, with 7-day lagged correlations and Toda-Yamamoto test results (v3, raw volume series). An updated analysis (share-transformed series, BH-FDR edge selection) is in progress and will replace this dataset.

Edge row format: `[source_v1, target_v1, ty_significant, ty_pvalue, ty_lag, r_lead1 ... r_lead7]`.

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
quantities and non-text metadata are included. The corrected series here
(share-transformed, BH-FDR-selected) supersede the raw-volume `data/v3/` data
used by the live viewer.

## Citation

If you use this software or data, please cite the accompanying article and this
archive (see [`CITATION.cff`](CITATION.cff)). The archived release is available
on Zenodo: **DOI: _to be added on release_**.

## License

- **Code** — the interactive viewer and the scripts under `replication/code/` —
  is released under the [MIT License](LICENSE).
- **Data** under `replication/data/` is released under
  [CC BY 4.0](replication/data/LICENSE).

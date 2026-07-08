# Replication package

Curated replication data and code for **"Qualitative Shock Mapping for
Quantitative Production Network Models: Evidence from Iran-Related Hostilities"**
(Kim & Erokhin, IIASA).

> **Raw article text is not redistributed** for copyright/licensing reasons.
> Only derived quantities and non-text metadata are provided, so the pipeline
> can be audited and re-run on independently collected text.

## Contents

```
replication/
├── data/                                  (CC BY 4.0 — see data/LICENSE)
│   ├── reference/
│   │   ├── sector_lookup_icio_v1_isic_rev4.csv    50 ICIO V1 sectors + ISIC Rev.4 codes/descriptions
│   │   └── icio_world_average_A_50x50.csv          world-average direct-requirement matrix (OECD ICIO, 2021 ed., 2018 ref. year)
│   ├── article_level/
│   │   ├── lexical_tfidf_probs_<lang>.csv           per-article 50-sector probabilities, lexical (TF-IDF) classifier + metadata (no raw text)
│   │   └── llm_probs_<lang>.csv                     per-article 50-sector probabilities, zero-shot LLM classifier
│   ├── daily_series/
│   │   ├── daily_indicators_lexical_<lang>.csv      daily sectoral indicators (uniform articles excluded)
│   │   ├── daily_share_lexical_<lang>.csv           within-day share-transformed series — the Section 6 specification of record
│   │   └── daily_share_llm_<lang>.csv               semantic-classifier share series (robustness comparison)
│   ├── networks/
│   │   ├── ty_edges_lexical_<lang>.csv              all 2,450 ordered pairs: Toda-Yamamoto p-values, lags, lead days, |r|, and the fdr_significant flag
│   │   ├── ty_edges_llm_<lang>.csv                  same, semantic classifier
│   │   ├── summary_lexical.csv                      network summary by language (Table 3)
│   │   └── summary_llm.csv
│   └── io_alignment/
│       ├── io_alignment_lexical_and_llm.csv         Fisher / QAP alignment vs the input-output structure
│       ├── io_alignment_llm.csv
│       └── io_alignment_factor_adjusted.csv
└── code/                                  (MIT — see ../LICENSE)
    ├── llm_classifier_prompt.txt           zero-shot LLM classification prompt (mapping hierarchy + 50-sector schema)
    ├── run_llm_full_batch.py               full-corpus LLM classification (batch API, prompt caching, post-processing)
    ├── run_v4_share_fdr.py                 lexical share-transform + BH-FDR lead-lag network (specification of record)
    ├── run_v5_llm_fdr.py                   semantic-classifier network (robustness)
    ├── phase3_io_alignment_*.py            input-output alignment tests (Fisher, QAP)
    ├── build_world_A.py                    world-average ICIO coefficient matrix
    ├── make_*.py / create_*.py             figure-generation scripts
    └── ...                                 (all analysis scripts)
```

`<lang>` ∈ {arabic, chinese, english, persian}.

## Mapping to the paper

| Paper element | File(s) |
|---|---|
| 50-sector scheme (§4) | `data/reference/sector_lookup_icio_v1_isic_rev4.csv` |
| Article-level exposure vectors p_i (§3.1, §5) | `data/article_level/*` |
| Daily indicators & within-day shares (§5, §6) | `data/daily_series/*` |
| Lead-lag networks, BH-FDR edges (§6, §7; Table 3, Figs 6–9) | `data/networks/ty_edges_*`, `summary_*` |
| Input-output alignment (§7, Fig 7) | `data/io_alignment/*`, `data/reference/icio_world_average_A_50x50.csv` |
| LLM classification (§3.1) | `code/llm_classifier_prompt.txt`, `code/run_llm_full_batch.py` |
| Figures | `code/make_*.py`, `code/create_*.py` |

## Reproducing

Scripts are plain Python (pandas, numpy, scikit-learn, statsmodels, matplotlib;
the LLM step uses the `anthropic` SDK). The LLM classifier reads its API key from
the `ANTHROPIC_API_KEY` environment variable — no key is stored in this
repository. Paths inside the scripts point to the authors' working tree and
should be adjusted to a local checkout.

## Licensing

- **Code** (`code/`, and the viewer at the repository root): MIT — see `../LICENSE`.
- **Data** (`data/`): CC BY 4.0 — see `data/LICENSE`.

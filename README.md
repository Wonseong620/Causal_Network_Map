# Causal Network Map

Interactive causal network visualization for article-level economic input-output sector exposure. The project maps 50 ICIO V1 sectors across Arabic, Chinese, English, and Persian Iran-related news data, then renders Toda-Yamamoto lead-lag relationships between sectors as a static HTML network viewer.

## What is included

- `index.html` - self-contained interactive network map for web deployment.
- `create_ty_v1_network_html.py` - builds versioned interactive HTML viewers from generated network CSVs.
- `create_ty_v1_networks.py` - generates Toda-Yamamoto network node, edge, daily occurrence, and PNG outputs from article-level IO probability matrices.
- `icio_sectors_with_isic_rev4_descriptions.csv` - lookup table for the 50 ICIO V1 sectors.
- `figures/causal network_v2/` - network CSVs used by the current HTML generator.
- `figures/causal network_v3/` - latest generated network CSV outputs.

Large raw article CSV inputs are intentionally excluded from the repository. The repository keeps the compact network data products and the static viewer needed to publish or inspect the map.

## View the map

Open `index.html` directly in a browser, or serve the repository with any static web server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Regenerate outputs

Install the Python dependencies:

```bash
pip install pandas numpy matplotlib networkx scipy statsmodels
```

Place the raw article-level matrix files in the repository root:

- `iran_ar_v3_with_io_matrix.csv`
- `iran_ch_v3_with_io_matrix.csv`
- `iran_en_v3_with_io_matrix.csv`
- `iran_pe_v3_with_io_matrix.csv`

Generate network CSV and PNG outputs:

```bash
python create_ty_v1_networks.py
```

Generate the interactive HTML outputs:

```bash
python create_ty_v1_network_html.py
```

## Deployment

The current web package is static. Upload the repository contents to a static host and keep `index.html` at the site root. No external CSV, JavaScript, CSS, or image assets are required by the deployed viewer.


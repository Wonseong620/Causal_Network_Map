TY v1 Interactive Network - Web Package

Contents
- index.html: self-contained static page for the interactive causal network.
- create_ty_v1_network_html.py: builds the interactive HTML viewer.
- create_ty_v1_networks.py: builds the Toda-Yamamoto network CSV/PNG outputs.
- figures/causal network_v2/: CSV files used by the current interactive HTML generator.
- figures/causal network_v3/: latest generated network CSV files.
- icio_sectors_with_isic_rev4_descriptions.csv: sector lookup table.

How to publish later
1. Upload the contents of this folder to any static web host.
2. Point the real domain to that host using the host's DNS instructions.
3. Keep index.html at the site root so the domain opens the visualization directly.

Notes
- No external CSV, image, JavaScript, or CSS files are required.
- The large raw article CSV inputs are excluded from GitHub; the included CSVs are
  the compact network data products related to the map.
- If your host supports custom domain files such as CNAME, add that file only after
  the final domain name is known.

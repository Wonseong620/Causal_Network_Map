# -*- coding: utf-8 -*-
"""Fig 7 from the hosted interactive viewer (https://wonseong620.github.io/Causal_Network_Map/).

Part 1 (selenium, headless Edge): set English / |r|>=0.92 / TY-significant,
lock focal node V1=26, capture the SVG under the three node-role modes.
Part 2 (matplotlib): compose the three captures into one 3-panel figure.

Run: python capture_fig7_viewer.py  (requires: pip install selenium)
"""
import io, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "figures" / "manuscript_v5"
CAPDIR = OUTDIR / "fig7_viewer_captures"
CAPDIR.mkdir(parents=True, exist_ok=True)
URL = "https://wonseong620.github.io/Causal_Network_Map/"
FOCAL = 26
CORR = 0.92

CAPTURES = [("both", "cap_hover.png"), ("lead", "cap_lead.png"), ("lag", "cap_lag.png")]

def capture():
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options
    from selenium.webdriver.common.by import By

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1500,1400")
    opts.add_argument("--force-device-scale-factor=2")
    opts.add_argument("--hide-scrollbars")
    driver = webdriver.Edge(options=opts)
    try:
        driver.get(URL)
        for _ in range(40):
            if driver.execute_script("return document.querySelectorAll('.node').length"):
                break
            time.sleep(0.5)
        assert driver.execute_script("return document.querySelectorAll('.node').length") == 50

        def set_control(cid, value):
            driver.execute_script(
                "const el = document.getElementById(arguments[0]);"
                "el.value = arguments[1];"
                "el.dispatchEvent(new Event('input', {bubbles: true}));",
                cid, str(value))

        set_control("language", "English"); time.sleep(2.5)
        set_control("corr", CORR); time.sleep(1.5)
        driver.execute_script(
            "document.querySelector('.node[data-v1=\"%d\"] circle')"
            ".dispatchEvent(new MouseEvent('click', {bubbles: true}));" % FOCAL)
        time.sleep(0.5)
        assert driver.execute_script(
            "return document.querySelector('.node.selected')?.dataset.v1") == str(FOCAL)

        svg_el = driver.find_element(By.ID, "network")
        for role, fname in CAPTURES:
            set_control("nodeRole", role); time.sleep(0.6)
            cnt = driver.execute_script("return document.getElementById('edgeCount').textContent")
            print(f"role={role}: visible edges = {cnt}")
            svg_el.screenshot(str(CAPDIR / fname))
    finally:
        driver.quit()

def compose():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    titles = [
        "(a) Hover / selected node — all incident edges (8)",
        "(b) Lead-only — sectors V1=26 precedes (3)",
        "(c) Lag-only — sectors that precede V1=26 (5)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.6))
    crop_top = 230  # drop the repeated per-panel title block
    for ax, (role, fname), title in zip(axes, CAPTURES, titles):
        img = mpimg.imread(CAPDIR / fname)
        ax.imshow(img[crop_top:, :, :])
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        ax.axis("off")
    fig.suptitle("Interactive web viewer: lead-lag interaction modes for V1=26 "
                 "(manufacture of other transport equipment) — English",
                 fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.5, 0.045,
             "Screenshots of the deployed viewer (wonseong620.github.io/Causal_Network_Map) with the "
             "selected node locked; filters |r| ≥ 0.92, max lead ≤ 7d, TY-significant edges, TY v3 data.",
             ha="center", fontsize=11.5, color="0.3")
    fig.text(0.5, 0.012,
             "Edge labels give the source sector's lead in days; non-neighbouring sectors are de-emphasised.",
             ha="center", fontsize=11.5, color="0.3")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = OUTDIR / "fig7_viewer_modes.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)

if __name__ == "__main__":
    if not all((CAPDIR / f).exists() for _, f in CAPTURES):
        capture()
    else:
        print("captures already present — composing only (delete folder to recapture)")
    compose()

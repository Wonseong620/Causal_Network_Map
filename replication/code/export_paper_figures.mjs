/* Export manuscript network figures from the viewer's paper mode.
 *
 * Renders ?export=paper views with headless Chromium and prints them to
 * vector PDF (fonts embedded) plus a PNG preview. Run from the repository
 * root with the viewer served locally:
 *
 *   python -m http.server 8766 &
 *   node replication/code/export_paper_figures.mjs [outDir] [baseUrl]
 *
 * Requires playwright (npm i -D playwright, or a global install).
 */
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const outDir = process.argv[2] ?? "figures/paper_export";
const base = process.argv[3] ?? "http://127.0.0.1:8766/";
mkdirSync(outDir, { recursive: true });

const COMMON = "export=paper&dataset=v5&ty=sig&corr=0&lead=7";
const FIGURES = [
  { name: "fig4_network_english", query: `${COMMON}&language=English&edgeMode=all` },
  { name: "fig5_network_4lang",   query: `${COMMON}&view=compare` },
  // Fig 7 panels: same focal sector, three lead-lag modes. Chinese V1 16
  // (rubber and plastic products) has the largest balanced degree in the FDR
  // network (11 lead / 9 lag edges), so all three modes are populated.
  { name: "fig7a_focus16_both",   query: `${COMMON}&language=Chinese&edgeMode=hover&node=16&nodeRole=both` },
  { name: "fig7b_focus16_lead",   query: `${COMMON}&language=Chinese&edgeMode=hover&node=16&nodeRole=lead` },
  { name: "fig7c_focus16_lag",    query: `${COMMON}&language=Chinese&edgeMode=hover&node=16&nodeRole=lag` },
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 1100 } });
page.on("pageerror", e => { console.error("page error:", e.message); process.exitCode = 1; });

for (const fig of FIGURES) {
  await page.goto(`${base}?${fig.query}`);
  await page.waitForFunction(() => document.querySelectorAll("#network .node").length >= 50);
  await page.waitForTimeout(300);
  await page.pdf({
    path: join(outDir, `${fig.name}.pdf`),
    width: "1000px", height: "1100px", printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
  });
  await page.screenshot({ path: join(outDir, `${fig.name}.png`) });
  console.log(`${fig.name} -> ${outDir}`);
}
await browser.close();

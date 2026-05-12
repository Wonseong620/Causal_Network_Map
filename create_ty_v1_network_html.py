"""Build an interactive HTML viewer for V1 Toda-Yamamoto networks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
V2_DIR = FIG_DIR / "causal network_v2"
HTML_DIR = FIG_DIR / "causal network_html"
HTML_FILE = HTML_DIR / "ty_v1_network_interactive.html"
LANGUAGES = ["Arabic", "Chinese", "English", "Persian"]
MAX_LEAD = 7


def corr_at_lag(source: pd.Series, target: pd.Series, lag: int) -> float:
    x = source.iloc[:-lag].to_numpy(dtype=float)
    y = target.iloc[lag:].to_numpy(dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not np.isfinite(r) else r


def language_payload(language: str) -> dict:
    slug = language.lower()
    nodes = pd.read_csv(V2_DIR / f"ty_v1_network_nodes_{slug}_v2.csv")
    ty_edges = pd.read_csv(V2_DIR / f"ty_v1_network_edges_{slug}_v2.csv")
    daily = pd.read_csv(V2_DIR / f"ty_v1_daily_occurrence_{slug}_v2.csv")

    ty_lookup = {
        (int(row.source), int(row.target)): {
            "p": float(row.ty_pvalue),
            "pLag": int(row.p_lag),
            "dmax": int(row.dmax),
        }
        for row in ty_edges.itertuples(index=False)
    }

    daily_cols = {int(col): pd.to_numeric(daily[col], errors="coerce").fillna(0.0) for col in map(str, range(1, 51))}
    edges = []
    for source in range(1, 51):
        for target in range(1, 51):
            if source == target:
                continue
            lag_corrs = [
                round(corr_at_lag(daily_cols[source], daily_cols[target], lag), 6)
                for lag in range(1, MAX_LEAD + 1)
            ]
            ty = ty_lookup.get((source, target))
            edges.append(
                {
                    "s": source,
                    "t": target,
                    "r": lag_corrs,
                    "ty": 1 if ty else 0,
                    "p": None if ty is None else round(ty["p"], 8),
                    "pLag": None if ty is None else ty["pLag"],
                    "dmax": None if ty is None else ty["dmax"],
                }
            )

    node_rows = []
    for row in nodes.itertuples(index=False):
        node_rows.append(
            {
                "v1": int(row.V1),
                "total": round(float(row.total_occurrence), 6),
                "active": int(row.active_days),
                "code": "" if pd.isna(row.Code) else str(row.Code),
                "industry": "" if pd.isna(row.Industry) else str(row.Industry),
            }
        )
    return {"nodes": node_rows, "edges": edges}


def build_html(data: dict, version: str, stagger_nodes: bool, hover_selection: bool) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    title_suffix = "V2" if version == "v2" else "V3"
    page_title = (
        "Interactive Causal Network Map"
        if version in {"v3", "v4"}
        else f"Interactive Toda-Yamamoto V1 Causal Network {title_suffix}"
    )
    extra_grid = " minmax(150px, 210px) minmax(150px, 210px)" if hover_selection else ""
    extra_control = """
      <label>Edge display
        <select id="edgeMode">
          <option value="hover">Hover / selected node</option>
          <option value="all">All</option>
        </select>
      </label>
      <label>Selected node role
        <select id="nodeRole">
          <option value="both">Lead + lag</option>
          <option value="lead">Lead only</option>
          <option value="lag">Lag only</option>
        </select>
      </label>""" if hover_selection else ""
    extra_hint = (
        " In hover mode, move over a node to show its incident edges; use selected-node role to show only edges it leads or only edges where it lags. Click a node to lock/unlock selection."
        if hover_selection
        else " Even V1 nodes are placed on an inner ring to reduce overlap."
    )
    extra_control_ref = (
        'edgeMode: document.getElementById("edgeMode"),\n'
        '  nodeRole: document.getElementById("nodeRole"),'
        if hover_selection
        else ""
    )
    active_state = "let hoverNode = null;\nlet selectedNode = null;" if hover_selection else ""
    edge_mode_logic = (
        "const edgeMode = controls.edgeMode.value;\n"
        "  const nodeRole = controls.nodeRole.value;\n"
        "  const activeNode = selectedNode || hoverNode;\n"
        "  const renderedEdges = edgeMode === \"hover\" ? visibleEdges.filter(e => {\n"
        "    if (!activeNode) return false;\n"
        "    if (nodeRole === \"lead\") return e.s === activeNode;\n"
        "    if (nodeRole === \"lag\") return e.t === activeNode;\n"
        "    return e.s === activeNode || e.t === activeNode;\n"
        "  }) : visibleEdges;\n"
        if hover_selection
        else "const renderedEdges = visibleEdges;\n"
    )
    node_events = (
        "g.addEventListener(\"mouseenter\", () => { hoverNode = node.v1; render(); });\n"
        "    g.addEventListener(\"mouseleave\", () => { hoverNode = null; render(); });\n"
        "    g.addEventListener(\"click\", () => { selectedNode = selectedNode === node.v1 ? null : node.v1; render(); });\n"
        if hover_selection
        else ""
    )
    radius_logic = (
        "const radius = v1 % 2 === 0 ? 320 : 400;"
        if stagger_nodes
        else "const radius = 380;"
    )
    selected_style = (
        'g.setAttribute("data-selected", selectedNode === node.v1 ? "1" : "0");'
        if hover_selection
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  <style>
    :root {{
      --blue: #0072bd;
      --ink: #111;
      --muted: #5d6670;
      --line: #d8dde3;
      --panel: #f7f9fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      padding: 18px 26px 14px;
      border-bottom: 1px solid var(--line);
      background: white;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(150px, 200px) minmax(220px, 1fr) minmax(180px, 280px) minmax(140px, 200px){extra_grid} minmax(140px, 190px);
      gap: 14px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: #27313b;
    }}
    select, input[type="range"] {{
      width: 100%;
    }}
    select {{
      height: 34px;
      border: 1px solid #b9c2cc;
      border-radius: 4px;
      background: white;
      font-size: 14px;
      padding: 0 8px;
    }}
    button {{
      height: 34px;
      border: 1px solid #8aa9c6;
      border-radius: 4px;
      background: var(--blue);
      color: white;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
    }}
    button:hover {{
      filter: brightness(.95);
    }}
    .value {{
      color: var(--blue);
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      min-height: 0;
    }}
    .canvas-wrap {{
      position: relative;
      min-height: 720px;
      overflow: hidden;
    }}
    svg {{
      width: 100%;
      height: 100%;
      min-height: 720px;
      display: block;
      background: white;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    .stat {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 9px 0;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }}
    .stat strong {{
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }}
    .legend {{
      margin-top: 18px;
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: #27313b;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 9px;
    }}
    .swatch-line {{
      width: 42px;
      height: 0;
      border-top: 5px solid var(--blue);
    }}
    .swatch-dot {{
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #808080;
    }}
    .hint {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .node circle {{
      stroke: white;
      stroke-width: 1.7;
      filter: drop-shadow(0 1px 2px rgba(0,0,0,.12));
    }}
    .node[data-selected="1"] circle {{
      stroke: #111;
      stroke-width: 3;
    }}
    .node text {{
      fill: white;
      font-size: 10px;
      font-weight: 800;
      text-anchor: middle;
      dominant-baseline: central;
      pointer-events: none;
    }}
    .edge {{
      fill: none;
      stroke: var(--blue);
      stroke-linecap: round;
      opacity: .43;
    }}
    .edge-label {{
      fill: var(--blue);
      font-size: 7px;
      font-weight: 700;
      paint-order: stroke;
      stroke: white;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    .title {{
      font-size: 22px;
      font-weight: 800;
      text-anchor: middle;
    }}
    .subtitle {{
      fill: #20262d;
      font-size: 13px;
      font-weight: 700;
      text-anchor: middle;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(17, 24, 39, .94);
      color: white;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 12px;
      line-height: 1.35;
      max-width: 300px;
      display: none;
      z-index: 5;
    }}
    @media (max-width: 980px) {{
      .controls {{ grid-template-columns: 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
<div class="app">
  <header>
    <h1>{page_title}</h1>
    <div class="controls">
      <label>Language
        <select id="language">
          <option>Arabic</option>
          <option>Chinese</option>
          <option>English</option>
          <option>Persian</option>
        </select>
      </label>
      <label>Correlation threshold: <span class="value" id="corrValue">0.70</span>
        <input id="corr" type="range" min="0" max="1" value="0.7" step="0.01" />
      </label>
      <label>Max lead day: <span class="value" id="leadValue">7</span>
        <input id="lead" type="range" min="1" max="7" value="7" step="1" />
      </label>
      <label>TY test significance
        <select id="ty">
          <option value="1">1</option>
          <option value="0">0</option>
        </select>
      </label>
      {extra_control}
      <label>Export
        <button id="download" type="button">Download PNG</button>
      </label>
    </div>
  </header>
  <main>
    <div class="canvas-wrap">
      <svg id="network" viewBox="0 0 1000 1000" role="img" aria-label="V1 causal network"></svg>
    </div>
    <aside>
      <div class="stat"><span>Visible edges</span><strong id="edgeCount">0</strong></div>
      <div class="stat"><span>Linked nodes</span><strong id="nodeCount">0</strong></div>
      <div class="stat"><span>Mean visible r</span><strong id="meanR">0.00</strong></div>
      <div class="stat"><span>Mean lead day</span><strong id="meanLead">0.0</strong></div>
      <div class="legend">
        <div class="legend-row"><span class="swatch-line"></span><span>Directed edge, thickness = correlation strength</span></div>
        <div class="legend-row"><span class="swatch-dot"></span><span>Node size = total soft occurrence</span></div>
        <div class="legend-row"><span class="value">+d</span><span>Edge label = source lead days</span></div>
      </div>
      <p class="hint">
        Nodes are fixed radially: V1=1 at the top, then clockwise to V1=50.
        When TY test significance is 1, only edges that passed the Toda-Yamamoto Wald test are shown.
        When it is 0, edges are filtered only by correlation threshold and max lead day.{extra_hint}
      </p>
    </aside>
  </main>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = {data_json};
const COLORS = ["#0072bd","#d95319","#edb120","#7e2f8e","#77ac30","#4dbeee","#a2142f"];
const svg = document.getElementById("network");
const tooltip = document.getElementById("tooltip");
const controls = {{
  language: document.getElementById("language"),
  corr: document.getElementById("corr"),
  lead: document.getElementById("lead"),
  ty: document.getElementById("ty"),
  {extra_control_ref}
  download: document.getElementById("download")
}};
{active_state}

function polarPosition(v1) {{
  const angle = Math.PI / 2 - 2 * Math.PI * (v1 - 1) / 50;
  {radius_logic}
  return {{ x: 500 + radius * Math.cos(angle), y: 505 - radius * Math.sin(angle) }};
}}

function bestEdge(edge, maxLead) {{
  let bestLag = 1;
  let bestR = -Infinity;
  for (let i = 0; i < maxLead; i++) {{
    const r = edge.r[i] || 0;
    if (r > bestR) {{
      bestR = r;
      bestLag = i + 1;
    }}
  }}
  return {{ lead: bestLag, r: bestR }};
}}

function curvedPath(a, b) {{
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const len = Math.max(1, Math.hypot(dx, dy));
  const bend = Math.min(55, len * 0.12);
  const cx = mx - dy / len * bend;
  const cy = my + dx / len * bend;
  return `M ${{a.x}} ${{a.y}} Q ${{cx}} ${{cy}} ${{b.x}} ${{b.y}}`;
}}

function trimEndpoints(a, b, sourceRadius, targetRadius) {{
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / len;
  const uy = dy / len;
  return {{
    a: {{ x: a.x + ux * (sourceRadius + 2), y: a.y + uy * (sourceRadius + 2) }},
    b: {{ x: b.x - ux * (targetRadius + 2), y: b.y - uy * (targetRadius + 2) }}
  }};
}}

function el(name, attrs = {{}}, text = "") {{
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}}

function render() {{
  const language = controls.language.value;
  const corrMin = Number(controls.corr.value);
  const maxLead = Number(controls.lead.value);
  const tyRequired = Number(controls.ty.value);
  document.getElementById("corrValue").textContent = corrMin.toFixed(2);
  document.getElementById("leadValue").textContent = String(maxLead);

  const payload = DATA[language];
  const nodes = payload.nodes;
  const maxTotal = Math.max(...nodes.map(d => d.total), 1);
  const radiusByV1 = new Map(nodes.map(d => [d.v1, 9 + 18 * Math.sqrt(d.total / maxTotal)]));
  const visibleEdges = [];
  for (const edge of payload.edges) {{
    if (tyRequired && edge.ty !== 1) continue;
    const best = bestEdge(edge, maxLead);
    if (best.r >= corrMin) visibleEdges.push({{...edge, lead: best.lead, bestR: best.r}});
  }}
  {edge_mode_logic}

  svg.replaceChildren();
  const defs = el("defs");
  const marker = el("marker", {{
    id: "arrow", viewBox: "0 0 10 10", refX: "10", refY: "5",
    markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse"
  }});
  marker.appendChild(el("path", {{ d: "M 0 0 L 10 5 L 0 10 z", fill: "#0072bd", opacity: "0.72" }}));
  defs.appendChild(marker);
  svg.appendChild(defs);

  svg.appendChild(el("text", {{ x: 500, y: 46, class: "title" }}, `Toda-Yamamoto Causal Network: ${{language}} V1 Sectors`));
  svg.appendChild(el("text", {{ x: 500, y: 76, class: "subtitle" }},
    `50 V1 sectors | ${{renderedEdges.length}} rendered / ${{visibleEdges.length}} matched edges | r >= ${{corrMin.toFixed(2)}} | max lead <= ${{maxLead}}d | TY=${{tyRequired}}`
  ));

  const edgeLayer = el("g");
  const labelLayer = el("g");
  for (const edge of renderedEdges) {{
    const rawA = polarPosition(edge.s);
    const rawB = polarPosition(edge.t);
    const trimmed = trimEndpoints(
      rawA,
      rawB,
      radiusByV1.get(edge.s) || 10,
      radiusByV1.get(edge.t) || 10
    );
    const a = trimmed.a;
    const b = trimmed.b;
    const path = el("path", {{
      d: curvedPath(a, b),
      class: "edge",
      "stroke-width": (0.8 + 5.4 * edge.bestR).toFixed(2),
      "marker-end": "url(#arrow)"
    }});
    path.addEventListener("mousemove", evt => showTip(evt, `V1 ${{edge.s}} -> V1 ${{edge.t}}<br>lead +${{edge.lead}}d<br>r=${{edge.bestR.toFixed(3)}}<br>TY=${{edge.ty}}${{edge.p !== null ? `<br>p=${{edge.p}}` : ""}}`));
    path.addEventListener("mouseleave", hideTip);
    edgeLayer.appendChild(path);

    const mid = {{ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }};
    labelLayer.appendChild(el("text", {{ x: mid.x, y: mid.y, class: "edge-label" }}, `+${{edge.lead}}d`));
  }}
  svg.appendChild(edgeLayer);
  svg.appendChild(labelLayer);

  const nodeLayer = el("g");
  for (const node of nodes) {{
    const p = polarPosition(node.v1);
    const radius = 9 + 18 * Math.sqrt(node.total / maxTotal);
    const g = el("g", {{ class: "node" }});
    {selected_style}
    g.appendChild(el("circle", {{ cx: p.x, cy: p.y, r: radius.toFixed(2), fill: COLORS[(node.v1 - 1) % COLORS.length] }}));
    g.appendChild(el("text", {{ x: p.x, y: p.y }}, String(node.v1)));
    {node_events}
    g.addEventListener("mousemove", evt => showTip(evt, `V1 ${{node.v1}} ${{node.code}}<br>${{node.industry}}<br>total=${{node.total.toFixed(2)}}<br>active days=${{node.active}}`));
    g.addEventListener("mouseleave", hideTip);
    nodeLayer.appendChild(g);
  }}
  svg.appendChild(nodeLayer);

  const linkedNodes = new Set();
  for (const edge of renderedEdges) {{
    linkedNodes.add(edge.s);
    linkedNodes.add(edge.t);
  }}
  const meanR = renderedEdges.length ? renderedEdges.reduce((s, e) => s + e.bestR, 0) / renderedEdges.length : 0;
  const meanLead = renderedEdges.length ? renderedEdges.reduce((s, e) => s + e.lead, 0) / renderedEdges.length : 0;
  document.getElementById("edgeCount").textContent = String(renderedEdges.length);
  document.getElementById("nodeCount").textContent = String(linkedNodes.size);
  document.getElementById("meanR").textContent = meanR.toFixed(2);
  document.getElementById("meanLead").textContent = meanLead.toFixed(1);
}}

function showTip(evt, html) {{
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = `${{evt.clientX + 14}}px`;
  tooltip.style.top = `${{evt.clientY + 14}}px`;
}}

function hideTip() {{
  tooltip.style.display = "none";
}}

for (const control of Object.values(controls)) control.addEventListener("input", render);
controls.download.addEventListener("click", downloadPng);

function downloadPng() {{
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = `
    .node circle {{ stroke: white; stroke-width: 1.7; }}
    .node text {{ fill: white; font-size: 10px; font-weight: 800; text-anchor: middle; dominant-baseline: central; font-family: Arial, Helvetica, sans-serif; }}
    .edge {{ fill: none; stroke: #0072bd; stroke-linecap: round; opacity: .43; }}
    .edge-label {{ fill: #0072bd; font-size: 7px; font-weight: 700; font-family: Arial, Helvetica, sans-serif; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }}
    .title {{ font-size: 22px; font-weight: 800; text-anchor: middle; font-family: Arial, Helvetica, sans-serif; }}
    .subtitle {{ fill: #20262d; font-size: 13px; font-weight: 700; text-anchor: middle; font-family: Arial, Helvetica, sans-serif; }}
  `;
  clone.insertBefore(style, clone.firstChild);
  const svgText = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([svgText], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {{
    const canvas = document.createElement("canvas");
    canvas.width = 2200;
    canvas.height = 2200;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    const pngUrl = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    const language = controls.language.value.toLowerCase();
    const corr = Number(controls.corr.value).toFixed(2).replace(".", "p");
    const lead = controls.lead.value;
    const ty = controls.ty.value;
    a.href = pngUrl;
    a.download = `ty_v1_network_{version}_${{language}}_corr${{corr}}_lead${{lead}}_ty${{ty}}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }};
  img.src = url;
}}
render();
</script>
</body>
</html>
"""


def main() -> None:
    data = {language: language_payload(language) for language in LANGUAGES}
    outputs = [
        (FIG_DIR / "causal network_html_v2" / "ty_v1_network_interactive_v2.html", "v2", True, False),
        (FIG_DIR / "causal network_html_v3" / "ty_v1_network_interactive_v3.html", "v3", True, True),
        (FIG_DIR / "causal network_html_v4" / "ty_v1_network_interactive_v4.html", "v4", True, True),
    ]
    for path, version, stagger_nodes, hover_selection in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_html(data, version, stagger_nodes, hover_selection), encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

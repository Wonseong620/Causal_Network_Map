/* Interactive Causal Network Map
 * Data: data/v5/<language>.json  (edge rows: [s, t, fdr, p, pLag, lead, r])
 *       data/v3/<language>.json  (edge rows: [s, t, ty, p, pLag, r1..r7], legacy)
 * Rendering strategy: the SVG scene is rebuilt only when filters change;
 * hover / selection updates toggle CSS classes on the existing DOM.
 * Add ?export=paper to the URL for a chrome-free, print-style view.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/* Sector groups: ICIO V1 sectors fall into seven contiguous blocks, so each
 * group is one arc of the ring. Colours are shared with the manuscript
 * (figures_code/qq_style.py) and were validated for colour-vision
 * deficiency on adjacent arcs. */
const GROUPS = [
  { key: "primary",   label: "Agriculture & mining",                    first: 1,  last: 8,  color: "#2ECC5B" },
  { key: "light_mfg", label: "Light & process manufacturing",           first: 9,  last: 19, color: "#9B59F5" },
  { key: "machinery", label: "Machinery & transport equipment",         first: 20, last: 27, color: "#FF7A1F" },
  { key: "utilities", label: "Utilities & construction",                first: 28, last: 30, color: "#3B6CFF" },
  { key: "trade",     label: "Trade, transport & logistics",            first: 31, last: 37, color: "#F0306B" },
  { key: "info_fin",  label: "Information, finance & business services", first: 38, last: 44, color: "#E0A400" },
  { key: "public",    label: "Public & social services",                first: 45, last: 50, color: "#00C2E0" },
];
const groupOf = v1 => GROUPS.find(g => v1 >= g.first && v1 <= g.last);
const sectorColor = v1 => groupOf(v1).color;

const EDGE_LABEL_MAX = 60;        // hide +Nd labels above this many visible edges
const DATASETS = {
  v5: { sigLabel: "BH-FDR 5% only" },
  v3: { sigLabel: "TY-significant only" },
};

const params = new URLSearchParams(location.search);
const PAPER = params.get("export") === "paper";
if (PAPER) document.body.classList.add("paper");
const CY = 505;                        // ring centre (titles live in the page header, not the SVG)

const svg = document.getElementById("network");
const tooltip = document.getElementById("tooltip");
const loading = document.getElementById("loading");
const controls = {
  dataset: document.getElementById("dataset"),
  language: document.getElementById("language"),
  corr: document.getElementById("corr"),
  lead: document.getElementById("lead"),
  ty: document.getElementById("ty"),
  edgeMode: document.getElementById("edgeMode"),
  nodeRole: document.getElementById("nodeRole"),
  view: document.getElementById("view"),
};
const stats = {
  edgeCount: document.getElementById("edgeCount"),
  nodeCount: document.getElementById("nodeCount"),
  meanR: document.getElementById("meanR"),
  meanLead: document.getElementById("meanLead"),
};

const cache = new Map();       // "<dataset>/<language>" -> payload
let hoverNode = null;
let selectedNode = null;
let scene = null;              // built DOM references for current filter set
let compareScene = null;       // per-panel DOM references in the 2x2 view

// ---------- data ----------
async function loadLanguage(language) {
  const ds = controls.dataset.value;
  const key = `${ds}/${language.toLowerCase()}`;
  if (cache.has(key)) return cache.get(key);
  loading.classList.add("on");
  try {
    const resp = await fetch(`data/${key}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${key}.json`);
    const raw = await resp.json();
    const edges = ds === "v5"
      ? raw.edges.map(row => ({ s: row[0], t: row[1], sig: row[2], p: row[3], pLag: row[4], lead: row[5], r: row[6] }))
      : raw.edges.map(row => ({ s: row[0], t: row[1], sig: row[2], p: row[3], pLag: row[4], rByLead: row.slice(5) }));
    const payload = { dataset: ds, meta: raw.meta, nodes: raw.nodes, edges };
    cache.set(key, payload);
    return payload;
  } finally {
    loading.classList.remove("on");
  }
}

// ---------- geometry ----------
function polarPosition(v1, cx = 500, cy = CY, scale = 1) {
  const angle = Math.PI / 2 - 2 * Math.PI * (v1 - 1) / 50;
  const radius = (v1 % 2 === 0 ? 320 : 400) * scale;
  return { x: cx + radius * Math.cos(angle), y: cy - radius * Math.sin(angle) };
}

const NODE_R0 = PAPER ? 15 : 12;   // paper export needs larger minimum nodes for print-size labels
const nodeRadius = (total, maxTotal, scale = 1) => (NODE_R0 + 24 * Math.sqrt(total / maxTotal)) * scale;

/* v3 only: pick the lag with the largest |r| within the lead window. */
function bestByLead(rByLead, maxLead) {
  let bestLag = 1, bestR = 0;
  for (let i = 0; i < maxLead; i++) {
    const r = rByLead[i] || 0;
    if (Math.abs(r) > Math.abs(bestR)) { bestR = r; bestLag = i + 1; }
  }
  return { lead: bestLag, r: bestR };
}

function controlPoint(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const bend = Math.min(55, len * 0.12);
  return { x: (a.x + b.x) / 2 - dy / len * bend, y: (a.y + b.y) / 2 + dx / len * bend };
}

/* Trim endpoints along the curve's tangent directions (a->c and c->b), so
 * arrowheads sit on the node boundary of the curved path rather than the
 * straight chord. */
function trimmedCurve(rawA, rawB, rA, rB) {
  const c = controlPoint(rawA, rawB);
  const tA = unit(rawA, c), tB = unit(c, rawB);
  const a = { x: rawA.x + tA.x * (rA + 2), y: rawA.y + tA.y * (rA + 2) };
  const b = { x: rawB.x - tB.x * (rB + 2), y: rawB.y - tB.y * (rB + 2) };
  return { a, b, c: controlPoint(a, b) };
}

function unit(p, q) {
  const dx = q.x - p.x, dy = q.y - p.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  return { x: dx / len, y: dy / len };
}

/* Arc spanning sectors v1First..v1Last clockwise, padded by 0.4 sector. */
function arcPath(cx, cy, r, v1First, v1Last) {
  const step = 2 * Math.PI / 50;
  const a0 = Math.PI / 2 - (v1First - 1 - 0.4) * step;
  const a1 = Math.PI / 2 - (v1Last - 1 + 0.4) * step;
  const p0 = { x: cx + r * Math.cos(a0), y: cy - r * Math.sin(a0) };
  const p1 = { x: cx + r * Math.cos(a1), y: cy - r * Math.sin(a1) };
  const large = (a0 - a1) > Math.PI ? 1 : 0;
  return `M ${p0.x.toFixed(1)} ${p0.y.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`;
}

// ---------- svg helpers ----------
function el(name, attrs = {}, text = "") {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text) node.textContent = text;
  return node;
}

function fmtP(p) {
  if (p === null || p === undefined) return "";
  return p < 0.001 ? "p&lt;0.001" : `p=${p.toFixed(3)}`;
}

/* Edge shade by source lead: darker = faster. */
const LEAD_SHADES = {
  fast: { pos: "#0072bd", neg: "#d95319" },
  mid:  { pos: "#4F9FD9", neg: "#EE8A5E" },
  slow: { pos: "#8ABEE6", neg: "#F2B08F" },
};
const leadClass = lead => (lead <= 2 ? "lead-fast" : lead <= 4 ? "lead-mid" : "lead-slow");
const markerId = (lead, neg) => `arrow-${leadClass(lead).slice(5)}-${neg ? "neg" : "pos"}`;

function arrowMarkers(defs) {
  const list = [];
  for (const [k, c] of Object.entries(LEAD_SHADES)) list.push([`arrow-${k}-pos`, c.pos], [`arrow-${k}-neg`, c.neg]);
  for (const [id, color] of list) {
    const m = el("marker", {
      id, viewBox: "0 0 10 10", refX: "10", refY: "5",
      markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse",
    });
    m.appendChild(el("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color, opacity: "0.72" }));
    defs.appendChild(m);
  }
}

function groupArcs(cx, cy, scale) {
  const g = el("g", { class: "group-arcs" });
  for (const grp of GROUPS) {
    g.appendChild(el("path", {
      d: arcPath(cx, cy, 448 * scale, grp.first, grp.last),
      stroke: grp.color, "stroke-width": (5 * scale).toFixed(1), fill: "none",
      "stroke-linecap": "round",
    }));
  }
  return g;
}

/* In-SVG legend used by the paper export (the page legend lives in <aside>). */
function svgLegend(y0) {
  const g = el("g", { class: "svg-legend" });
  GROUPS.forEach((grp, i) => {
    const col = Math.floor(i / 4), row = i % 4;
    const x = 40 + col * 480, y = y0 + row * 26;
    g.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: grp.color, stroke: "white", "stroke-width": 1 }));
    g.appendChild(el("text", { x: x + 16, y: y + 5, class: "legend-text" }, `${grp.label} (${grp.first}–${grp.last})`));
  });
  const y = y0 + 3 * 26, x = 40 + 480;
  const rows = [["lead-fast", "Lead 1\u20132 d"], ["lead-mid", "3\u20134 d"], ["lead-slow", "5\u20137 d"]];
  let cx = x - 6;
  for (const [cls, label] of rows) {
    g.appendChild(el("path", { d: `M ${cx} ${y} L ${cx + 30} ${y}`, class: `edge ${cls}`, style: "opacity:1", "stroke-width": 3 }));
    g.appendChild(el("text", { x: cx + 36, y: y + 5, class: "legend-text" }, label));
    cx += label.length > 6 ? 150 : 100;
  }
  g.appendChild(el("path", { d: `M ${cx} ${y} L ${cx + 30} ${y}`, class: "edge neg lead-fast", style: "opacity:1", "stroke-width": 3 }));
  g.appendChild(el("text", { x: cx + 36, y: y + 5, class: "legend-text" }, "Negative"));
  return g;
}

// ---------- filtering ----------
function currentFilters() {
  return {
    corrMin: Number(controls.corr.value),
    maxLead: Number(controls.lead.value),
    sig: controls.ty.value,           // "sig" | "p05" | "all"
  };
}

function visibleEdges(payload, f) {
  const out = [];
  for (const edge of payload.edges) {
    if (f.sig === "sig" && edge.sig !== 1) continue;
    if (f.sig === "p05" && !(edge.p !== null && edge.p < 0.05)) continue;
    let lead, r;
    if (payload.dataset === "v5") {
      lead = edge.lead; r = edge.r;
      if (lead > f.maxLead) continue;
    } else {
      ({ lead, r } = bestByLead(edge.rByLead, f.maxLead));
    }
    if (Math.abs(r) >= f.corrMin) out.push({ ...edge, lead, bestR: r });
  }
  return out;
}

function sigLabel(f) {
  if (f.sig === "sig") return controls.dataset.value === "v5" ? "BH-FDR 5%" : "TY-significant";
  return f.sig === "p05" ? "raw p<0.05" : "all correlations";
}

// ---------- scene construction (filter changes only) ----------
function buildScene(payload, language) {
  const f = currentFilters();
  const edges = visibleEdges(payload, f);
  const nodes = payload.nodes;
  const maxTotal = Math.max(...nodes.map(d => d.total), 1);
  const radiusByV1 = new Map(nodes.map(d => [d.v1, nodeRadius(d.total, maxTotal)]));
  const showLabels = edges.length <= EDGE_LABEL_MAX;

  svg.replaceChildren();
  const defs = el("defs");
  arrowMarkers(defs);
  svg.appendChild(defs);
  svg.setAttribute("viewBox", PAPER ? "0 0 1000 1100" : "0 0 1000 1000");
  svg.classList.toggle("paper", PAPER);

  setViewHead(`${language} media lead\u2013lag network by sector`, "");

  const emptyMsg = el("text", { x: 500, y: CY, class: "empty-msg" });
  svg.appendChild(emptyMsg);
  svg.appendChild(groupArcs(500, CY, 1));

  const edgeLayer = el("g");
  const labelLayer = el("g");
  const edgeEls = [];
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const geo = trimmedCurve(
      polarPosition(e.s), polarPosition(e.t),
      radiusByV1.get(e.s) || 10, radiusByV1.get(e.t) || 10);
    const neg = e.bestR < 0;
    const path = el("path", {
      d: `M ${geo.a.x} ${geo.a.y} Q ${geo.c.x} ${geo.c.y} ${geo.b.x} ${geo.b.y}`,
      class: `edge ${leadClass(e.lead)}${neg ? " neg" : ""}`,
      "stroke-width": (0.8 + 5.4 * Math.abs(e.bestR)).toFixed(2),
      "marker-end": `url(#${markerId(e.lead, neg)})`,
      "data-edge": i,
    });
    const label = el("text", {
      x: (geo.a.x + geo.b.x) / 2, y: (geo.a.y + geo.b.y) / 2,
      class: neg ? "edge-label neg" : "edge-label",
    }, `+${e.lead}d`);
    edgeLayer.appendChild(path);
    labelLayer.appendChild(label);
    edgeEls.push({ path, label, e });
  }
  svg.appendChild(edgeLayer);
  svg.appendChild(labelLayer);

  const nodeLayer = el("g");
  const nodeEls = new Map();
  for (const node of nodes) {
    const p = polarPosition(node.v1);
    const g = el("g", { class: "node", "data-v1": node.v1, tabindex: "0", role: "button" });
    g.setAttribute("aria-label", `Sector V1 ${node.v1}: ${node.industry}`);
    g.appendChild(el("circle", {
      cx: p.x, cy: p.y, r: (radiusByV1.get(node.v1) || 10).toFixed(2),
      fill: sectorColor(node.v1),
    }));
    g.appendChild(el("text", { x: p.x, y: p.y }, String(node.v1)));
    nodeLayer.appendChild(g);
    nodeEls.set(node.v1, g);
  }
  svg.appendChild(nodeLayer);
  const focalLabel = el("text", { class: "focal-label hidden" });
  svg.appendChild(focalLabel);
  if (PAPER) svg.appendChild(svgLegend(975));

  scene = { payload, language, edges, edgeEls, nodeEls, emptyMsg, focalLabel, filters: f, showLabels };
  updateInteraction();
}

// ---------- interaction updates (hover / selection: class toggles only) ----------
function updateInteraction() {
  if (!scene) return;
  const mode = controls.edgeMode.value;
  const role = controls.nodeRole.value;
  const active = selectedNode ?? hoverNode;

  let rendered = 0, sumR = 0, sumLead = 0;
  const linked = new Set();
  for (const { path, label, e } of scene.edgeEls) {
    let show;
    if (mode === "all") {
      show = true;
    } else if (!active) {
      show = false;
    } else if (role === "lead") {
      show = e.s === active;
    } else if (role === "lag") {
      show = e.t === active;
    } else {
      show = e.s === active || e.t === active;
    }
    path.classList.toggle("hidden", !show);
    // Labels: always on for a focused node, otherwise only when the graph is sparse.
    label.classList.toggle("hidden", !show || (mode === "all" && !scene.showLabels));
    if (show) {
      rendered++;
      sumR += Math.abs(e.bestR);
      sumLead += e.lead;
      linked.add(e.s); linked.add(e.t);
    }
  }

  for (const [v1, g] of scene.nodeEls) {
    g.classList.toggle("selected", v1 === selectedNode);
    const dim = active !== null && active !== undefined
      && mode === "hover" && rendered > 0 && v1 !== active && !linked.has(v1);
    g.classList.toggle("dim", dim);
  }

  const f = scene.filters;
  setViewHead(null,
    `${rendered} of ${scene.edges.length} matched edges shown \u00b7 |r| \u2265 ${f.corrMin.toFixed(2)} \u00b7 ` +
    `lead \u2264 ${f.maxLead} d \u00b7 ${sigLabel(f)} \u00b7 ${scene.payload.meta.dataset}`);

  if (scene.edges.length === 0) {
    scene.emptyMsg.textContent = "No edges match the current filters - lower the |r| threshold or relax the significance filter.";
  } else if (mode === "hover" && !active) {
    scene.emptyMsg.textContent = "Hover or click a sector node to explore its lead-lag edges.";
  } else {
    scene.emptyMsg.textContent = "";
  }

  stats.edgeCount.textContent = String(rendered);
  stats.nodeCount.textContent = String(linked.size);
  stats.meanR.textContent = rendered ? (sumR / rendered).toFixed(2) : "0.00";
  stats.meanLead.textContent = rendered ? (sumLead / rendered).toFixed(1) : "0.0";

  placeFocalLabel(active);
  renderNodeCard(active, rendered);
  const visible = scene.edgeEls.filter(({ path }) => !path.classList.contains("hidden")).map(x => x.e);
  renderGroupMatrix(visible);
  renderEdgeTable(visible);
}

/* Short sector name beside the focused node, placed outside the ring so
 * it never covers edges. */
function placeFocalLabel(active) {
  const lbl = scene.focalLabel;
  if (!lbl) return;
  const n = active ? scene.payload.nodes.find(d => d.v1 === active) : null;
  lbl.classList.toggle("hidden", !n);
  if (!n) return;
  // Outside the ring; near the left/right extremes the label would leave the
  // viewBox, so it sits above or below the node instead.
  const p = polarPosition(n.v1, 500, CY, 1.16);
  const dx = p.x - 500, dy = p.y - CY;
  const side = Math.abs(dx) > 340;
  lbl.setAttribute("x", side ? polarPosition(n.v1, 500, CY, 1).x : p.x);
  lbl.setAttribute("y", side ? polarPosition(n.v1, 500, CY, 1).y + (dy >= 0 ? 52 : -44) : p.y + 4);
  lbl.setAttribute("text-anchor", side ? "middle" : (dx >= 0 ? "start" : "end"));
  lbl.textContent = `${n.v1} ${shortName(n.industry, 24)}`;
}

function shortName(industry, max = 28) {
  const t = industry.split(";")[0].replace(/^Manufacture of /, "").replace(/^Activities of /, "");
  return t.length <= max ? t : t.slice(0, max).replace(/\s+\S*$/, "") + "\u2026";
}

function setViewHead(title, sub) {
  if (title !== null) document.getElementById("viewTitle").textContent = title;
  if (sub !== null) document.getElementById("viewSub").textContent = sub;
}

function renderNodeCard(active, rendered) {
  const body = document.getElementById("nodeCardBody");
  if (!body) return;
  const n = active && scene ? scene.payload.nodes.find(d => d.v1 === active) : null;
  if (!n) {
    body.innerHTML = controls.view.value === "compare"
      ? "Switch to a single language to inspect a sector."
      : "Hover or click a node on the ring. Click again to unlock.";
    return;
  }
  const g = groupOf(n.v1);
  let out = 0, inn = 0;
  for (const e of scene.edges) { if (e.s === n.v1) out++; if (e.t === n.v1) inn++; }
  body.innerHTML =
    `<div class="name">V1 ${n.v1} \u00b7 ${n.code}</div>` +
    `<div>${n.industry}</div>` +
    `<div class="grp"><i style="background:${g.color}"></i>${g.label}</div>` +
    `<dl><dt>Total exposure</dt><dd>${n.total.toFixed(1)}</dd>` +
    `<dt>Active days</dt><dd>${n.active}</dd>` +
    `<dt>Leads</dt><dd>${out} sector${out === 1 ? "" : "s"}</dd>` +
    `<dt>Lags</dt><dd>${inn} sector${inn === 1 ? "" : "s"}</dd>` +
    `<dt>Locked</dt><dd>${selectedNode === n.v1 ? "yes" : "no"}</dd></dl>`;
}

/* Sortable table of the visible edges, with CSV export. */
const edgeSort = { key: "r", asc: false };
let tableEdges = [];
const EDGE_COLS = [
  ["s", "From"], ["t", "To"], ["lead", "Lead d"], ["r", "r"], ["p", "p"],
];
function renderEdgeTable(edges) {
  const table = document.getElementById("edgeTable");
  if (!table) return;
  tableEdges = edges;
  document.getElementById("edgeTableCount").textContent = edges.length ? `(${edges.length})` : "";
  const nodes = scene?.payload.nodes ?? [];
  const nameOf = v1 => nodes.find(d => d.v1 === v1)?.industry.split(";")[0] ?? "";
  const k = edgeSort.key;
  const val = e => (k === "r" ? Math.abs(e.bestR) : k === "p" ? (e.p ?? 1) : e[k]);
  const rows = [...edges].sort((a, b) => (val(a) - val(b)) * (edgeSort.asc ? 1 : -1));
  let html = "<thead><tr>" + EDGE_COLS.map(([key, label]) =>
    `<th data-key="${key}" class="${key === k ? "sorted" + (edgeSort.asc ? " asc" : "") : ""}">${label}</th>`).join("") + "</tr></thead><tbody>";
  if (!rows.length) {
    html += `<tr><td class="empty" colspan="${EDGE_COLS.length}">No visible edges</td></tr>`;
  }
  for (const e of rows.slice(0, 400)) {
    const cell = v1 => `<td title="${nameOf(v1)}"><i class="dot" style="background:${sectorColor(v1)}"></i>${v1}</td>`;
    html += `<tr>${cell(e.s)}${cell(e.t)}<td>+${e.lead}</td>` +
      `<td class="${e.bestR < 0 ? "neg" : ""}">${e.bestR.toFixed(3)}</td>` +
      `<td>${e.p === null || e.p === undefined ? "" : e.p < 0.001 ? "&lt;0.001" : e.p.toFixed(3)}</td></tr>`;
  }
  if (rows.length > 400) html += `<tr><td class="empty" colspan="${EDGE_COLS.length}">Showing 400 of ${rows.length}; download CSV for all</td></tr>`;
  table.innerHTML = html + "</tbody>";
}
document.getElementById("edgeTable").addEventListener("click", evt => {
  const th = evt.target.closest("th[data-key]");
  if (!th) return;
  const key = th.dataset.key;
  if (edgeSort.key === key) edgeSort.asc = !edgeSort.asc;
  else { edgeSort.key = key; edgeSort.asc = key === "s" || key === "t" || key === "lead" || key === "p"; }
  renderEdgeTable(tableEdges);
});
document.getElementById("edgeCsv").addEventListener("click", () => {
  if (!scene) { showToast("Switch to a single language first"); return; }
  const nodes = scene.payload.nodes;
  const nameOf = v1 => nodes.find(d => d.v1 === v1)?.industry ?? "";
  const q = v => `"${String(v).replace(/"/g, '""')}"`;
  const lines = ["source_v1,source_industry,target_v1,target_industry,lead_days,pearson_r,ty_pvalue,ty_lag,significant"];
  for (const e of tableEdges) {
    lines.push([e.s, q(nameOf(e.s)), e.t, q(nameOf(e.t)), e.lead, e.bestR.toFixed(4),
      e.p ?? "", e.pLag ?? "", e.sig ? 1 : 0].join(","));
  }
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" }));
  download(url, exportName("csv").replace(/^network_/, "edges_"));
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast(`${tableEdges.length} edges exported`);
});

/* 7x7 group-to-group edge counts for the currently visible edges. */
function renderGroupMatrix(edges) {
  const box = document.getElementById("groupMatrix");
  if (!box) return;
  const idx = new Map(GROUPS.map((g, i) => [g.key, i]));
  const m = GROUPS.map(() => GROUPS.map(() => 0));
  for (const e of edges) m[idx.get(groupOf(e.s).key)][idx.get(groupOf(e.t).key)]++;
  const max = Math.max(1, ...m.flat());
  const abbr = ["Agr", "LtM", "Mch", "Utl", "Trd", "Inf", "Pub"];
  let html = "<table class=\"gm\"><thead><tr><th></th>" +
    GROUPS.map((g, j) => `<th style="color:${g.color}" title="${g.label}">${abbr[j]}</th>`).join("") +
    "</tr></thead><tbody>";
  GROUPS.forEach((g, i) => {
    html += `<tr><th style="color:${g.color}" title="${g.label}">${abbr[i]}</th>` +
      m[i].map(v => {
        const a = v ? (0.12 + 0.78 * v / max).toFixed(2) : 0;
        return `<td style="background:rgba(0,114,189,${a})${v / max > 0.55 ? ";color:#fff" : ""}">${v || ""}</td>`;
      }).join("") + "</tr>";
  });
  box.innerHTML = html + "</tbody></table>";
}

// ---------- compare (2x2 small multiples) ----------
async function buildCompare() {
  const f = currentFilters();
  const langs = ["Arabic", "Chinese", "English", "Persian"];
  const payloads = await Promise.all(langs.map(loadLanguage));

  svg.replaceChildren();
  const defs = el("defs");
  arrowMarkers(defs);
  svg.appendChild(defs);
  svg.setAttribute("viewBox", PAPER ? "0 0 1000 1100" : "0 0 1000 1000");
  svg.classList.toggle("paper", PAPER);
  setViewHead("Four-language lead\u2013lag comparison",
    `|r| \u2265 ${f.corrMin.toFixed(2)} \u00b7 lead \u2264 ${f.maxLead} d \u00b7 ${sigLabel(f)} \u00b7 ${payloads[0].meta.dataset}`);

  const top = 70;
  const centers = [[260, top + 215], [740, top + 215], [260, top + 690], [740, top + 690]];
  const scale = 0.46;
  const panelLabels = ["(a)", "(b)", "(c)", "(d)"];
  const panels = [];
  langs.forEach((lang, i) => {
    const payload = payloads[i];
    const [cx, cy] = centers[i];
    const edges = visibleEdges(payload, f);
    const maxTotal = Math.max(...payload.nodes.map(d => d.total), 1);
    const g = el("g");
    g.appendChild(el("text", { x: cx, y: cy - 226, class: "panel-title" },
      `${PAPER ? panelLabels[i] + " " : ""}${lang} (${edges.length} edges)`));
    g.appendChild(groupArcs(cx, cy, scale));
    if (edges.length === 0) {
      g.appendChild(el("text", { x: cx, y: cy + 5, class: "empty-msg" }, "No selected edges"));
    }
    const radiusByV1 = new Map(payload.nodes.map(d => [d.v1, nodeRadius(d.total, maxTotal, scale)]));
    const panel = { lang, payload, edgeEls: [], nodeEls: new Map() };
    for (const e of edges) {
      const a = polarPosition(e.s, cx, cy, scale);
      const b = polarPosition(e.t, cx, cy, scale);
      const geo = trimmedCurve(a, b, radiusByV1.get(e.s), radiusByV1.get(e.t));
      const neg = e.bestR < 0;
      const path = el("path", {
        d: `M ${geo.a.x} ${geo.a.y} Q ${geo.c.x} ${geo.c.y} ${geo.b.x} ${geo.b.y}`,
        class: `edge ${leadClass(e.lead)}${neg ? " neg" : ""}`,
        "stroke-width": (0.5 + 2.6 * Math.abs(e.bestR)).toFixed(2),
        "marker-end": `url(#${markerId(e.lead, neg)})`,
      });
      g.appendChild(path);
      panel.edgeEls.push({ path, e });
    }
    for (const node of payload.nodes) {
      const p = polarPosition(node.v1, cx, cy, scale);
      const ng = el("g", { class: "node small", "data-v1": node.v1, tabindex: "0", role: "button" });
      ng.setAttribute("aria-label", `${lang}: sector V1 ${node.v1}: ${node.industry}`);
      ng.appendChild(el("circle", {
        cx: p.x, cy: p.y, r: radiusByV1.get(node.v1).toFixed(2),
        fill: sectorColor(node.v1),
      }));
      ng.appendChild(el("text", { x: p.x, y: p.y }, String(node.v1)));
      g.appendChild(ng);
      panel.nodeEls.set(node.v1, ng);
    }
    panels.push(panel);
    svg.appendChild(g);
  });
  if (PAPER) svg.appendChild(svgLegend(985));
  scene = null;
  compareScene = { panels };
  const box = document.getElementById("groupMatrix");
  if (box) box.innerHTML = "";
  updateCompareHighlight();
  tableEdges = [];
  document.getElementById("edgeTableCount").textContent = "";
  document.getElementById("edgeTable").innerHTML = `<tbody><tr><td class="empty">Switch to a single language to list edges</td></tr></tbody>`;
  stats.edgeCount.textContent = "-";
  stats.nodeCount.textContent = "-";
  stats.meanR.textContent = "-";
  stats.meanLead.textContent = "-";
}

/* Same sector highlighted in all four panels: incident edges stay, the
 * rest fade; the role filter applies as in the single view. */
function updateCompareHighlight() {
  if (!compareScene) return;
  const active = selectedNode ?? hoverNode;
  const role = controls.nodeRole.value;
  const counts = [];
  for (const panel of compareScene.panels) {
    const linked = new Set();
    let out = 0, inn = 0;
    for (const { path, e } of panel.edgeEls) {
      let show = true;
      if (active) {
        show = role === "lead" ? e.s === active : role === "lag" ? e.t === active : (e.s === active || e.t === active);
      }
      path.classList.toggle("hidden", !show);
      if (show && active) { linked.add(e.s); linked.add(e.t); if (e.s === active) out++; if (e.t === active) inn++; }
    }
    for (const [v1, g] of panel.nodeEls) {
      g.classList.toggle("selected", v1 === selectedNode);
      g.classList.toggle("focus", active !== null && active !== undefined && v1 === active);
      g.classList.toggle("dim", !!active && v1 !== active && !linked.has(v1));
    }
    counts.push({ lang: panel.lang, out, inn });
  }
  renderCompareCard(active, counts);
}

function renderCompareCard(active, counts) {
  const body = document.getElementById("nodeCardBody");
  if (!body) return;
  const n = active ? compareScene.panels[0].payload.nodes.find(d => d.v1 === active) : null;
  if (!n) {
    body.innerHTML = "Hover or click a node in any panel to highlight that sector in all four languages.";
    return;
  }
  const g = groupOf(n.v1);
  body.innerHTML =
    `<div class="name">V1 ${n.v1} \u00b7 ${n.code}</div>` +
    `<div>${n.industry}</div>` +
    `<div class="grp"><i style="background:${g.color}"></i>${g.label}</div>` +
    `<dl>` + counts.map(c => `<dt>${c.lang}</dt><dd>leads ${c.out} \u00b7 lags ${c.inn}</dd>`).join("") +
    `<dt>Locked</dt><dd>${selectedNode === n.v1 ? "yes" : "no"}</dd></dl>`;
}

// ---------- render orchestration ----------
function syncDatasetUi() {
  const ds = controls.dataset.value;
  document.getElementById("version").textContent = `v2.1.3 \u00b7 data ${ds}`;
  controls.ty.querySelector('option[value="sig"]').textContent = DATASETS[ds].sigLabel;
  document.getElementById("dataNote").textContent = ds === "v5"
    ? "Data: v5 - within-day share-transformed series; edges selected by Benjamini-Hochberg FDR (5%) over all 2,450 ordered sector pairs. Specification of record for the manuscript."
    : "Data: v3 - raw volume series with a fixed |r| cutoff. Kept for comparison; superseded by v5.";
}

async function render({ rebuild = true } = {}) {
  syncDatasetUi();
  if (controls.view.value === "compare") {
    if (rebuild || !compareScene) await buildCompare();
    else updateCompareHighlight();
    if (compareScene) fillSectorList(compareScene.panels[0].payload.nodes);
    return;
  }
  compareScene = null;
  const language = controls.language.value;
  document.title = `${language} Media Lead-Lag by Sectors`;
  document.getElementById("corrValue").textContent = Number(controls.corr.value).toFixed(2);
  document.getElementById("leadValue").textContent = controls.lead.value;
  if (rebuild || !scene || scene.language !== language) {
    const payload = await loadLanguage(language);
    fillSectorList(payload.nodes);
    buildScene(payload, language);
  } else {
    updateInteraction();
  }
}

// ---------- event delegation ----------
svg.addEventListener("pointerover", evt => {
  const nodeG = evt.target.closest(".node[data-v1]");
  if (nodeG && compareScene) {
    hoverNode = Number(nodeG.dataset.v1);
    updateCompareHighlight();
    const n = compareScene.panels[0].payload.nodes.find(d => d.v1 === hoverNode);
    if (n) showTip(evt, `<b>V1 ${n.v1} ${n.code}</b><br>${n.industry}<br><i>${groupOf(n.v1).label}</i>`);
    return;
  }
  if (nodeG) {
    hoverNode = Number(nodeG.dataset.v1);
    updateInteraction();
    const n = scene?.payload.nodes.find(d => d.v1 === hoverNode);
    if (n) showTip(evt, `<b>V1 ${n.v1} ${n.code}</b><br>${n.industry}<br><i>${groupOf(n.v1).label}</i><br>total=${n.total.toFixed(2)}<br>active days=${n.active}`);
    return;
  }
  const pathEl = evt.target.closest("[data-edge]");
  if (pathEl && scene) {
    const e = scene.edgeEls[Number(pathEl.dataset.edge)].e;
    const sig = controls.dataset.value === "v5" ? "BH-FDR significant" : "TY-significant";
    showTip(evt, `V1 ${e.s} &rarr; V1 ${e.t}<br>lead +${e.lead}d<br>r=${e.bestR.toFixed(3)}<br>` +
      `${e.sig ? sig : "not " + sig}${e.p !== null ? "<br>" + fmtP(e.p) : ""}`);
  }
});
svg.addEventListener("pointermove", evt => {
  if (tooltip.style.display === "block") moveTip(evt);
});
svg.addEventListener("pointerout", evt => {
  const nodeG = evt.target.closest(".node[data-v1]");
  if (nodeG && !nodeG.contains(evt.relatedTarget)) {
    hoverNode = null;
    if (compareScene) updateCompareHighlight(); else updateInteraction();
  }
  hideTip();
});
svg.addEventListener("click", evt => {
  const nodeG = evt.target.closest(".node[data-v1]");
  if (!nodeG) return;
  const v1 = Number(nodeG.dataset.v1);
  selectedNode = selectedNode === v1 ? null : v1;
  syncSectorBox();
  if (compareScene) updateCompareHighlight(); else updateInteraction();
});
svg.addEventListener("keydown", evt => {
  const nodeG = evt.target.closest(".node[data-v1]");
  if (!nodeG || (evt.key !== "Enter" && evt.key !== " ")) return;
  evt.preventDefault();
  const v1 = Number(nodeG.dataset.v1);
  selectedNode = selectedNode === v1 ? null : v1;
  syncSectorBox();
  if (compareScene) updateCompareHighlight(); else updateInteraction();
});

function syncSectorBox() {
  const box = document.getElementById("sectorFind");
  const nodes = scene?.payload.nodes ?? compareScene?.panels[0].payload.nodes;
  if (!box || !nodes) return;
  const n = selectedNode ? nodes.find(d => d.v1 === selectedNode) : null;
  box.value = n ? `${n.v1} \u2014 ${n.industry.split(";")[0]}` : "";
}

function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  moveTip(evt);
}
function moveTip(evt) {
  tooltip.style.left = `${evt.clientX + 14}px`;
  tooltip.style.top = `${evt.clientY + 14}px`;
}
function hideTip() { tooltip.style.display = "none"; }

// filter controls rebuild the scene; display controls only retoggle classes
for (const id of ["dataset", "language", "corr", "lead", "ty", "view"]) {
  controls[id].addEventListener("input", () => {
    selectedNode = null;
    render({ rebuild: true });
  });
}
for (const id of ["edgeMode", "nodeRole"]) {
  controls[id].addEventListener("input", () => render({ rebuild: false }));
}

// ---------- export ----------
const EXPORT_CSS = `
    text { font-family: Inter, Arial, Helvetica, sans-serif; }
    .node circle { stroke: white; stroke-width: 1.7; }
    .node text { fill: white; font-size: 13px; font-weight: 900; text-anchor: middle; dominant-baseline: central; paint-order: stroke; stroke: rgba(0,0,0,.35); stroke-width: 1.6px; stroke-linejoin: round; }
    .node.small text { font-size: 7px; }
    .node.dim circle, .node.dim text { opacity: .22; }
    .node.selected circle { stroke: #111; stroke-width: 3; }
    .edge { fill: none; stroke: #0072bd; stroke-linecap: round; opacity: .43; }
    .edge.neg { stroke: #d95319; stroke-dasharray: 6 4; }
    .edge.lead-mid { stroke: #4F9FD9; } .edge.lead-slow { stroke: #8ABEE6; }
    .edge.neg.lead-mid { stroke: #EE8A5E; } .edge.neg.lead-slow { stroke: #F2B08F; }
    .edge.hidden, .edge-label.hidden { display: none; }
    .edge-label { fill: #0072bd; font-size: 8px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }
    .edge-label.neg { fill: #d95319; }
    .title { font-size: 18px; font-weight: 800; text-anchor: middle; }
    .subtitle { fill: #20262d; font-size: 13px; font-weight: 700; text-anchor: middle; }
    .datastamp { fill: #8a929b; font-size: 10px; text-anchor: middle; }
    .empty-msg { fill: #5d6670; font-size: 15px; font-weight: 700; text-anchor: middle; }
    .panel-title { font-size: 14px; font-weight: 800; text-anchor: middle; }
    .legend-text { fill: #111; font-size: 12px; }
    .focal-label { fill: #111; font-size: 13px; font-weight: 800; paint-order: stroke; stroke: white; stroke-width: 4px; stroke-linejoin: round; }
    .focal-label.hidden { display: none; }
    svg.paper .focal-label { font-size: 17px; }
    svg.paper .node text { font-size: 15px; }
    svg.paper .node.small text { font-size: 9px; }
    svg.paper .edge-label { font-size: 11px; }
    svg.paper .panel-title { font-size: 20px; }
    svg.paper .legend-text { font-size: 17px; }
    svg.paper .empty-msg { font-size: 18px; }
  `;

function exportName(ext) {
  const view = controls.view.value === "compare" ? "compare" : controls.language.value.toLowerCase();
  const corr = Number(controls.corr.value).toFixed(2).replace(".", "p");
  return `network_${controls.dataset.value}_${view}_corr${corr}_lead${controls.lead.value}_${controls.ty.value}.${ext}`;
}

function serializeSvg() {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", SVG_NS);
  const style = document.createElementNS(SVG_NS, "style");
  style.textContent = EXPORT_CSS;
  clone.insertBefore(style, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

function download(href, name) {
  const a = document.createElement("a");
  a.href = href; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

document.getElementById("downloadSvg").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([serializeSvg()], { type: "image/svg+xml;charset=utf-8" }));
  download(url, exportName("svg"));
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

document.getElementById("download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([serializeSvg()], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  img.onload = () => {
    const [, , vw, vh] = svg.getAttribute("viewBox").split(" ").map(Number);
    const canvas = document.createElement("canvas");
    canvas.width = 2200;
    canvas.height = Math.round(2200 * vh / vw);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    download(canvas.toDataURL("image/png"), exportName("png"));
  };
  img.src = url;
});

// ---------- sector search ----------
const sectorFind = document.getElementById("sectorFind");
let sectorListFilled = false;
function fillSectorList(nodes) {
  if (sectorListFilled) return;
  const dl = document.getElementById("sectorList");
  for (const n of nodes) {
    const o = document.createElement("option");
    o.value = `${n.v1} \u2014 ${n.industry.split(";")[0]}`;
    dl.appendChild(o);
  }
  sectorListFilled = true;
}
function findSector(text) {
  const nodes = scene?.payload.nodes ?? compareScene?.panels[0].payload.nodes;
  if (!nodes) return null;
  const t = text.trim().toLowerCase();
  if (!t) return null;
  const m = t.match(/^(\d{1,2})\b/);
  if (m) { const v = Number(m[1]); if (v >= 1 && v <= 50) return v; }
  const hit = nodes.find(n => n.industry.toLowerCase().includes(t) || n.code.toLowerCase() === t);
  return hit ? hit.v1 : null;
}
async function selectSector(v1) {
  selectedNode = v1;
  syncSectorBox();
  if (compareScene) updateCompareHighlight(); else updateInteraction();
  scene?.nodeEls.get(v1)?.focus({ preventScroll: true });
}
sectorFind.addEventListener("change", () => {
  const v1 = findSector(sectorFind.value);
  if (v1) selectSector(v1);
  else if (sectorFind.value.trim()) showToast("No sector matches");
});
function clearSelection() {
  sectorFind.value = ""; selectedNode = null;
  if (compareScene) updateCompareHighlight(); else updateInteraction();
}
sectorFind.addEventListener("keydown", evt => { if (evt.key === "Escape") clearSelection(); });
document.getElementById("sectorClear").addEventListener("click", clearSelection);

// ---------- about dialog ----------
const aboutDlg = document.getElementById("about");
document.getElementById("aboutOpen").addEventListener("click", () => aboutDlg.showModal());
document.getElementById("aboutClose").addEventListener("click", () => aboutDlg.close());
aboutDlg.addEventListener("click", evt => { if (evt.target === aboutDlg) aboutDlg.close(); });
if (params.get("about") === "1") aboutDlg.showModal();

// ---------- share link / small screens ----------
function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("on");
  setTimeout(() => t.classList.remove("on"), 1800);
}
document.getElementById("copyLink").addEventListener("click", async () => {
  const q = new URLSearchParams();
  for (const [id, c] of Object.entries(controls)) q.set(id, c.value);
  if (selectedNode) q.set("node", String(selectedNode));
  const url = `${location.origin}${location.pathname}?${q}`;
  try { await navigator.clipboard.writeText(url); showToast("Link copied"); }
  catch { prompt("Copy this link:", url); }
});
if (window.matchMedia("(max-width: 820px)").matches && !PAPER) {
  document.getElementById("filters").classList.add("collapsed");
  document.getElementById("menuToggle").setAttribute("aria-expanded", "false");
}
document.getElementById("menuToggle").addEventListener("click", evt => {
  const nav = document.getElementById("filters");
  const open = nav.classList.toggle("collapsed") === false;
  evt.currentTarget.setAttribute("aria-expanded", String(open));
});

// URL presets, e.g. ?dataset=v5&language=English&corr=0&lead=7&ty=sig&edgeMode=all&view=single
// Add &node=26&nodeRole=lead to lock a focal sector, &export=paper for a print view.
for (const id of Object.keys(controls)) {
  if (params.has(id)) controls[id].value = params.get(id);
}
if (params.has("node")) selectedNode = Number(params.get("node"));
render({ rebuild: true });

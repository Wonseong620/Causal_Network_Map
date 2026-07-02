/* Interactive Causal Network Map
 * Data: data/v3/<language>.json  (edge rows: [s, t, ty, p, pLag, r1..r7])
 * Rendering strategy: the SVG scene is rebuilt only when filters change;
 * hover / selection updates toggle CSS classes on the existing DOM.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30", "#4dbeee", "#a2142f"];
const DATA_VERSION = "TY v3 (raw volume series)";

const svg = document.getElementById("network");
const tooltip = document.getElementById("tooltip");
const loading = document.getElementById("loading");
const controls = {
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

const cache = new Map();       // language -> payload
let hoverNode = null;
let selectedNode = null;
let scene = null;              // built DOM references for current filter set

// ---------- data ----------
async function loadLanguage(language) {
  const key = language.toLowerCase();
  if (cache.has(key)) return cache.get(key);
  loading.classList.add("on");
  try {
    const resp = await fetch(`data/v3/${key}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${key}.json`);
    const raw = await resp.json();
    const payload = {
      meta: raw.meta,
      nodes: raw.nodes,
      edges: raw.edges.map(row => ({
        s: row[0], t: row[1], ty: row[2], p: row[3], pLag: row[4],
        r: row.slice(5),
      })),
    };
    cache.set(key, payload);
    return payload;
  } finally {
    loading.classList.remove("on");
  }
}

// ---------- geometry ----------
function polarPosition(v1, cx = 500, cy = 505, scale = 1) {
  const angle = Math.PI / 2 - 2 * Math.PI * (v1 - 1) / 50;
  const radius = (v1 % 2 === 0 ? 320 : 400) * scale;
  return { x: cx + radius * Math.cos(angle), y: cy - radius * Math.sin(angle) };
}

/* Pick the lag with the largest |r| within the lead window — matches the
 * pipeline's best_positive_lead_corr (abs comparison), unlike the previous
 * viewer which compared raw r and silently dropped negative-correlation
 * edges. */
function bestEdge(edge, maxLead) {
  let bestLag = 1;
  let bestR = 0;
  for (let i = 0; i < maxLead; i++) {
    const r = edge.r[i] || 0;
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

function arrowMarkers(defs) {
  for (const [id, color] of [["arrow", "#0072bd"], ["arrowNeg", "#d95319"]]) {
    const m = el("marker", {
      id, viewBox: "0 0 10 10", refX: "10", refY: "5",
      markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse",
    });
    m.appendChild(el("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color, opacity: "0.72" }));
    defs.appendChild(m);
  }
}

// ---------- filtering ----------
function currentFilters() {
  return {
    corrMin: Number(controls.corr.value),
    maxLead: Number(controls.lead.value),
    tyRequired: Number(controls.ty.value),
  };
}

function visibleEdges(payload, f) {
  const out = [];
  for (const edge of payload.edges) {
    if (f.tyRequired && edge.ty !== 1) continue;
    const best = bestEdge(edge, f.maxLead);
    if (Math.abs(best.r) >= f.corrMin) {
      out.push({ ...edge, lead: best.lead, bestR: best.r });
    }
  }
  return out;
}

// ---------- scene construction (filter changes only) ----------
function buildScene(payload, language) {
  const f = currentFilters();
  const edges = visibleEdges(payload, f);
  const nodes = payload.nodes;
  const maxTotal = Math.max(...nodes.map(d => d.total), 1);
  const radiusByV1 = new Map(nodes.map(d => [d.v1, 9 + 18 * Math.sqrt(d.total / maxTotal)]));

  svg.replaceChildren();
  const defs = el("defs");
  arrowMarkers(defs);
  svg.appendChild(defs);

  svg.appendChild(el("text", { x: 500, y: 42, class: "title" },
    `${language} Media Lead-Lag by Sectors`));
  const sub = el("text", { x: 500, y: 70, class: "subtitle" });
  svg.appendChild(sub);
  svg.appendChild(el("text", { x: 500, y: 90, class: "datastamp" },
    `Data: ${DATA_VERSION} - updated analysis in progress`));

  const emptyMsg = el("text", { x: 500, y: 505, class: "empty-msg" });
  svg.appendChild(emptyMsg);

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
      class: neg ? "edge neg" : "edge",
      "stroke-width": (0.8 + 5.4 * Math.abs(e.bestR)).toFixed(2),
      "marker-end": neg ? "url(#arrowNeg)" : "url(#arrow)",
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
      fill: COLORS[(node.v1 - 1) % COLORS.length],
    }));
    g.appendChild(el("text", { x: p.x, y: p.y }, String(node.v1)));
    nodeLayer.appendChild(g);
    nodeEls.set(node.v1, g);
  }
  svg.appendChild(nodeLayer);

  scene = { payload, language, edges, edgeEls, nodeEls, sub, emptyMsg, filters: f };
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
    label.classList.toggle("hidden", !show);
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
  scene.sub.textContent =
    `50 V1 sectors | ${rendered} rendered / ${scene.edges.length} matched edges | ` +
    `|r| >= ${f.corrMin.toFixed(2)} | max lead <= ${f.maxLead}d | ` +
    (f.tyRequired ? "TY-significant" : "all correlations");

  if (scene.edges.length === 0) {
    scene.emptyMsg.textContent = "No edges match the current filters - lower the |r| threshold or set TY to all correlations.";
  } else if (mode === "hover" && !active) {
    scene.emptyMsg.textContent = "Hover or click a sector node to explore its lead-lag edges.";
  } else {
    scene.emptyMsg.textContent = "";
  }

  stats.edgeCount.textContent = String(rendered);
  stats.nodeCount.textContent = String(linked.size);
  stats.meanR.textContent = rendered ? (sumR / rendered).toFixed(2) : "0.00";
  stats.meanLead.textContent = rendered ? (sumLead / rendered).toFixed(1) : "0.0";
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
  svg.appendChild(el("text", { x: 500, y: 34, class: "title" },
    "Four-Language Lead-Lag Comparison"));
  svg.appendChild(el("text", { x: 500, y: 58, class: "subtitle" },
    `|r| >= ${f.corrMin.toFixed(2)} | max lead <= ${f.maxLead}d | ` +
    (f.tyRequired ? "TY-significant" : "all correlations")));
  svg.appendChild(el("text", { x: 500, y: 76, class: "datastamp" },
    `Data: ${DATA_VERSION} - updated analysis in progress`));

  const centers = [[260, 300], [740, 300], [260, 760], [740, 760]];
  const scale = 0.52;
  langs.forEach((lang, i) => {
    const payload = payloads[i];
    const [cx, cy] = centers[i];
    const edges = visibleEdges(payload, f);
    const maxTotal = Math.max(...payload.nodes.map(d => d.total), 1);
    const g = el("g");
    g.appendChild(el("text", { x: cx, y: cy - 225, class: "panel-title" },
      `${lang} (${edges.length} edges)`));
    for (const e of edges) {
      const a = polarPosition(e.s, cx, cy, scale);
      const b = polarPosition(e.t, cx, cy, scale);
      const geo = trimmedCurve(a, b, 5, 5);
      const neg = e.bestR < 0;
      g.appendChild(el("path", {
        d: `M ${geo.a.x} ${geo.a.y} Q ${geo.c.x} ${geo.c.y} ${geo.b.x} ${geo.b.y}`,
        class: neg ? "edge neg" : "edge",
        "stroke-width": (0.5 + 2.6 * Math.abs(e.bestR)).toFixed(2),
        "marker-end": neg ? "url(#arrowNeg)" : "url(#arrow)",
      }));
    }
    for (const node of payload.nodes) {
      const p = polarPosition(node.v1, cx, cy, scale);
      const r = 3.5 + 8 * Math.sqrt(node.total / maxTotal);
      g.appendChild(el("circle", {
        cx: p.x, cy: p.y, r: r.toFixed(2),
        fill: COLORS[(node.v1 - 1) % COLORS.length],
        stroke: "white", "stroke-width": "0.8",
      }));
    }
    svg.appendChild(g);
  });
  scene = null;
  stats.edgeCount.textContent = "-";
  stats.nodeCount.textContent = "-";
  stats.meanR.textContent = "-";
  stats.meanLead.textContent = "-";
}

// ---------- render orchestration ----------
async function render({ rebuild = true } = {}) {
  if (controls.view.value === "compare") {
    await buildCompare();
    return;
  }
  const language = controls.language.value;
  document.title = `${language} Media Lead-Lag by Sectors`;
  document.getElementById("corrValue").textContent = Number(controls.corr.value).toFixed(2);
  document.getElementById("leadValue").textContent = controls.lead.value;
  if (rebuild || !scene || scene.language !== language) {
    const payload = await loadLanguage(language);
    buildScene(payload, language);
  } else {
    updateInteraction();
  }
}

// ---------- event delegation ----------
svg.addEventListener("pointerover", evt => {
  const nodeG = evt.target.closest(".node");
  if (nodeG) {
    hoverNode = Number(nodeG.dataset.v1);
    updateInteraction();
    const n = scene?.payload.nodes.find(d => d.v1 === hoverNode);
    if (n) showTip(evt, `<b>V1 ${n.v1} ${n.code}</b><br>${n.industry}<br>total=${n.total.toFixed(2)}<br>active days=${n.active}`);
    return;
  }
  const pathEl = evt.target.closest("[data-edge]");
  if (pathEl && scene) {
    const e = scene.edgeEls[Number(pathEl.dataset.edge)].e;
    showTip(evt, `V1 ${e.s} &rarr; V1 ${e.t}<br>lead +${e.lead}d<br>r=${e.bestR.toFixed(3)}<br>` +
      `${e.ty ? "TY-significant" : "not TY-significant"}${e.p !== null ? "<br>" + fmtP(e.p) : ""}`);
  }
});
svg.addEventListener("pointermove", evt => {
  if (tooltip.style.display === "block") moveTip(evt);
});
svg.addEventListener("pointerout", evt => {
  const nodeG = evt.target.closest(".node");
  if (nodeG && !nodeG.contains(evt.relatedTarget)) {
    hoverNode = null;
    updateInteraction();
  }
  hideTip();
});
svg.addEventListener("click", evt => {
  const nodeG = evt.target.closest(".node");
  if (!nodeG) return;
  const v1 = Number(nodeG.dataset.v1);
  selectedNode = selectedNode === v1 ? null : v1;
  updateInteraction();
});
svg.addEventListener("keydown", evt => {
  const nodeG = evt.target.closest(".node");
  if (!nodeG || (evt.key !== "Enter" && evt.key !== " ")) return;
  evt.preventDefault();
  const v1 = Number(nodeG.dataset.v1);
  selectedNode = selectedNode === v1 ? null : v1;
  updateInteraction();
});

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
for (const id of ["language", "corr", "lead", "ty", "view"]) {
  controls[id].addEventListener("input", () => {
    selectedNode = null;
    render({ rebuild: true });
  });
}
for (const id of ["edgeMode", "nodeRole"]) {
  controls[id].addEventListener("input", () => render({ rebuild: false }));
}

// ---------- PNG export ----------
document.getElementById("download").addEventListener("click", () => {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", SVG_NS);
  const style = document.createElementNS(SVG_NS, "style");
  style.textContent = `
    text { font-family: Arial, Helvetica, sans-serif; }
    circle { stroke: white; stroke-width: 1.7; }
    .node text { fill: white; font-size: 10px; font-weight: 800; text-anchor: middle; dominant-baseline: central; }
    .node.dim circle, .node.dim text { opacity: .22; }
    .edge { fill: none; stroke: #0072bd; stroke-linecap: round; opacity: .43; }
    .edge.neg { stroke: #d95319; stroke-dasharray: 6 4; }
    .edge.hidden, .edge-label.hidden { display: none; }
    .edge-label { fill: #0072bd; font-size: 7px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }
    .edge-label.neg { fill: #d95319; }
    .title { font-size: 18px; font-weight: 800; text-anchor: middle; }
    .subtitle { fill: #20262d; font-size: 13px; font-weight: 700; text-anchor: middle; }
    .datastamp { fill: #8a929b; font-size: 10px; text-anchor: middle; }
    .empty-msg { fill: #5d6670; font-size: 15px; font-weight: 700; text-anchor: middle; }
    .panel-title { font-size: 13px; font-weight: 800; text-anchor: middle; }
  `;
  clone.insertBefore(style, clone.firstChild);
  const svgText = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 2200;
    canvas.height = 2200;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    const a = document.createElement("a");
    const view = controls.view.value === "compare" ? "compare" : controls.language.value.toLowerCase();
    const corr = Number(controls.corr.value).toFixed(2).replace(".", "p");
    a.href = canvas.toDataURL("image/png");
    a.download = `ty_v1_network_${view}_corr${corr}_lead${controls.lead.value}_ty${controls.ty.value}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  img.src = url;
});

render({ rebuild: true });

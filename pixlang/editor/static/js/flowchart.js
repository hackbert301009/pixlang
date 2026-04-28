// pixlang/editor/static/js/flowchart.js
// Renders a pixlang AST graph (nodes + edges from /api/parse) as an SVG diagram.

const SVG_NS = "http://www.w3.org/2000/svg";

// Node dimensions and spacing
const NODE_W  = 176;
const NODE_H  = 32;
const V_GAP   = 44;   // vertical gap between nodes
const PADDING = 18;

// Category → fill colour (semi-transparent) and stroke colour
const CATEGORY_COLOURS = {
  IO:          { fill: "#2563eb22", stroke: "#3b82f6" },
  Geometry:    { fill: "#7c3aed22", stroke: "#a78bfa" },
  Color:       { fill: "#db277722", stroke: "#f472b6" },
  Threshold:   { fill: "#d9770622", stroke: "#fbbf24" },
  Filter:      { fill: "#0891b222", stroke: "#22d3ee" },
  Morphology:  { fill: "#05966922", stroke: "#34d399" },
  Analysis:    { fill: "#ea580c22", stroke: "#fb923c" },
  Composition: { fill: "#4f46e522", stroke: "#818cf8" },
  Annotation:  { fill: "#65a30d22", stroke: "#a3e635" },
  Control:     { fill: "#ca8a0422", stroke: "#fde047" },
  Variable:    { fill: "#92400e22", stroke: "#d97706" },
  Assert:      { fill: "#c2410c22", stroke: "#f97316" },
  ROI:         { fill: "#6d28d922", stroke: "#c084fc" },
  Include:     { fill: "#37415122", stroke: "#6b7280" },
  Other:       { fill: "#37415122", stroke: "#6b7280" },
};

function getColour(category) {
  return CATEGORY_COLOURS[category] || CATEGORY_COLOURS.Other;
}

// ── SVG helpers ───────────────────────────────────────────────────────────────

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function svgText(content, x, y, extra = {}) {
  const t = svgEl("text", {
    x, y,
    "text-anchor": "middle",
    "dominant-baseline": "middle",
    ...extra,
  });
  t.textContent = truncate(content, 24);
  return t;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

// ── Layout ────────────────────────────────────────────────────────────────────
// Simple top-down layout: assign (x, y) by index order (linear pipeline).

function layoutNodes(nodes) {
  const positions = {};
  // Centre each node horizontally within the SVG width (computed later)
  // For now store the rank (y-index); final x is computed in render.
  nodes.forEach((node, i) => {
    positions[node.id] = { rank: i };
  });
  return positions;
}

// ── Render ────────────────────────────────────────────────────────────────────

export function renderFlowchart(graphData, svgEl_, onNodeClick) {
  const graph = document.getElementById("flow-graph");
  graph.innerHTML = "";

  const { nodes = [], edges = [], error } = graphData;

  if (error || nodes.length === 0) {
    const msg = error || "Write code to see the flow diagram";
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", "50%");
    t.setAttribute("y", "40");
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#484f58");
    t.setAttribute("font-size", "11");
    t.setAttribute("font-family", "monospace");
    t.textContent = truncate(msg, 40);
    graph.appendChild(t);
    svgEl_.setAttribute("height", "80");
    return;
  }

  // Compute layout
  const containerW = svgEl_.parentElement
    ? svgEl_.parentElement.clientWidth || 240
    : 240;
  const cx = containerW / 2;

  const positions = {};
  nodes.forEach((node, i) => {
    const x = cx - NODE_W / 2;
    const y = PADDING + i * (NODE_H + V_GAP);
    positions[node.id] = { x, y };
  });

  const totalH = PADDING + nodes.length * (NODE_H + V_GAP) + PADDING;
  svgEl_.setAttribute("height", Math.max(totalH, 80));
  svgEl_.setAttribute("viewBox", `0 0 ${containerW} ${totalH}`);

  // ── Draw edges (behind nodes) ─────────────────────────────────────────────
  for (const edge of edges) {
    const from = positions[edge.from];
    const to   = positions[edge.to];
    if (!from || !to) continue;

    const group = document.createElementNS(SVG_NS, "g");
    group.classList.add("flow-edge");

    const x1 = from.x + NODE_W / 2;
    const y1 = from.y + NODE_H;
    const x2 = to.x   + NODE_W / 2;
    const y2 = to.y;
    const isLoopBack = to.y < from.y;

    if (isLoopBack) {
      // Loop-back edge: arc around the left side
      const path = document.createElementNS(SVG_NS, "path");
      const cpx  = x1 - 55;
      path.setAttribute("d",
        `M${x1},${y1} C${cpx},${y1} ${cpx},${y2} ${x2},${y2}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#4a5568");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("marker-end", "url(#arrowhead)");
      path.setAttribute("stroke-dasharray", "5,3");
      group.appendChild(path);
    } else {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1);
      line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      line.setAttribute("stroke", "#30363d");
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("marker-end", "url(#arrowhead)");
      group.appendChild(line);
    }

    if (edge.label) {
      const midX  = (x1 + x2) / 2 + 6;
      const midY  = (y1 + y2) / 2;
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", midX);
      label.setAttribute("y", midY);
      label.setAttribute("font-size", "9");
      label.setAttribute("fill", "#6e7681");
      label.setAttribute("font-family", "monospace");
      label.textContent = edge.label;
      group.appendChild(label);
    }

    graph.appendChild(group);
  }

  // ── Draw nodes ────────────────────────────────────────────────────────────
  for (const node of nodes) {
    const pos    = positions[node.id];
    if (!pos) continue;
    const colour = getColour(node.category);
    const group  = document.createElementNS(SVG_NS, "g");
    group.classList.add("flow-node");
    group.setAttribute("data-line", node.line);
    group.setAttribute("data-id",   node.id);

    const { x, y } = pos;

    if (node.type === "if_header" || node.type === "repeat_header") {
      // Diamond shape
      const cx2 = x + NODE_W / 2, cy2 = y + NODE_H / 2;
      const hw   = NODE_W / 2,    hh   = NODE_H / 2 + 4;
      const poly = svgEl("polygon", {
        points:         `${cx2},${cy2 - hh} ${cx2 + hw},${cy2} ${cx2},${cy2 + hh} ${cx2 - hw},${cy2}`,
        fill:           colour.fill,
        stroke:         colour.stroke,
        "stroke-width": "1.5",
        class:          "flow-poly",
      });
      group.appendChild(poly);
    } else if (node.type === "include") {
      // Dashed rectangle
      const rect = svgEl("rect", {
        x, y, width: NODE_W, height: NODE_H, rx: "4",
        fill:              colour.fill,
        stroke:            colour.stroke,
        "stroke-width":    "1.5",
        "stroke-dasharray": "5,3",
        class:             "flow-rect",
      });
      group.appendChild(rect);
    } else if (node.type === "roi_header" || node.type === "roi_footer") {
      // Wide rounded rect (extra padding)
      const rect = svgEl("rect", {
        x: x - 4, y, width: NODE_W + 8, height: NODE_H, rx: "6",
        fill:           colour.fill,
        stroke:         colour.stroke,
        "stroke-width": "2",
        class:          "flow-rect",
      });
      group.appendChild(rect);
    } else if (node.type === "assert") {
      // Rounded rect with thicker border
      const rect = svgEl("rect", {
        x, y, width: NODE_W, height: NODE_H, rx: "4",
        fill:           colour.fill,
        stroke:         colour.stroke,
        "stroke-width": "2",
        class:          "flow-rect",
      });
      group.appendChild(rect);
    } else {
      // Standard rectangle
      const rect = svgEl("rect", {
        x, y, width: NODE_W, height: NODE_H, rx: "4",
        fill:           colour.fill,
        stroke:         colour.stroke,
        "stroke-width": "1.5",
        class:          "flow-rect",
      });
      group.appendChild(rect);
    }

    // Label text
    const textEl = svgEl("text", {
      x:                  x + NODE_W / 2,
      y:                  y + NODE_H / 2,
      "text-anchor":      "middle",
      "dominant-baseline": "middle",
      "font-family":      "monospace",
      "font-size":        "10",
      fill:               "#e6edf3",
    });
    textEl.textContent = truncate(node.label, 24);
    group.appendChild(textEl);

    // Click → jump to line
    if (node.line && onNodeClick) {
      group.style.cursor = "pointer";
      group.addEventListener("click", () => onNodeClick(node.line));
      group.setAttribute("title", `Line ${node.line}`);
    }

    graph.appendChild(group);
  }
}

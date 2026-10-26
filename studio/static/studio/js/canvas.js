// The node canvas: draws nodes and edges, and handles dragging, connecting, panning and zooming.

import { edges, outPorts, ROUTERS } from './graph.js';
import { h, icon, iconButton, svg } from './ui.js';

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 2;
const FIT_MIN_ZOOM = 0.5;

export class Canvas {
  constructor(editor) {
    this.editor = editor;
    this.pan = { x: 120, y: 260 };
    this.zoom = 1;
    this.gesture = null; // {kind: 'drag' | 'pan' | 'connect', ...}
    this.queued = false;

    this.edgeLayer = svg('svg', { class: 'edges', width: '1', height: '1' });
    this.nodeLayer = h('div', { class: 'nodes' });
    this.edgeButton = h('button', {
      type: 'button', class: 'edge-delete', title: 'Remove connection (Delete)', hidden: true,
      onClick: () => this.editor.deleteSelection(),
    }, icon('x'));
    this.viewport = h('div', { class: 'viewport' }, this.edgeLayer, this.nodeLayer, this.edgeButton);
    this.hint = h('div', { class: 'canvas-hint', hidden: true },
      'Click a node type on the left to add it after the selected node, or drag it here. ',
      'Drag from a node\'s right dot to another node to connect them.');
    this.root = h('div', { class: 'canvas', tabindex: '0', 'aria-label': 'Workflow canvas' },
      this.viewport, this.hint, this.controls());
    this.bind();
    this.applyTransform();
  }

  controls() {
    return h('div', { class: 'canvas-controls' },
      iconButton('zoomIn', 'Zoom in', () => this.zoomBy(1.2)),
      iconButton('zoomOut', 'Zoom out', () => this.zoomBy(1 / 1.2)),
      iconButton('fit', 'Fit to screen', () => this.fit()),
      iconButton('layout', 'Arrange nodes automatically', () => this.editor.autoArrange()));
  }

  // --- Rendering ---------------------------------------------------------------------

  scheduleRender() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      this.render();
    });
  }

  render() {
    const { dsl } = this.editor;
    this.nodeLayer.replaceChildren(...Object.entries(dsl.nodes).map(([id, node]) => this.renderNode(id, node)));
    this.hint.hidden = Object.keys(dsl.nodes).length > 2;
    this.drawEdges();
  }

  renderNode(id, node) {
    const { types, selection, errorsByNode, runState } = this.editor;
    const info = types[node.type] || { title: node.type };
    const pos = this.editor.position(id);
    const errors = errorsByNode[id];
    const run = runState[id];
    const classes = ['node', `t-${node.type}`];
    if (selection?.type === 'node' && selection.id === id) classes.push('selected');
    if (errors) classes.push('has-error');
    if (run) classes.push(`run-${run.status}`);

    let badge = null;
    if (run?.status === 'running') badge = h('span', { class: 'node-badge running', title: 'Running' }, h('span', { class: 'spinner' }));
    else if (run?.status === 'done') badge = h('span', { class: 'node-badge done', title: 'Finished' }, `${run.ms} ms`);
    else if (run?.status === 'failed') badge = h('span', { class: 'node-badge failed', title: run.error || 'Failed' }, icon('alert'));
    else if (errors) badge = h('span', { class: 'node-badge error', title: errors.join('\n') }, icon('alert'));

    const el = h('div', { class: classes.join(' '), dataset: { id }, style: { left: `${pos.x}px`, top: `${pos.y}px` } },
      node.type === 'begin' ? null : h('div', { class: 'port in', title: 'Input' }),
      h('div', { class: 'node-head' },
        h('span', { class: 'node-icon' }, icon(node.type)),
        h('div', { class: 'node-titles' }, h('div', { class: 'node-id' }, id), h('div', { class: 'node-type' }, info.title)),
        badge),
      this.summary(node));

    if (ROUTERS.has(node.type)) {
      el.append(h('div', { class: 'routes' }, outPorts(node).map((port) => h('div', { class: `route ${port.key === 'default' ? 'else' : ''}` },
        h('span', { class: 'route-label', title: port.label }, port.label),
        h('div', { class: `port out ${port.target ? 'connected' : ''}`, dataset: { port: port.key }, title: 'Drag to connect' })))));
    } else {
      const [port] = outPorts(node);
      el.append(h('div', { class: `port out ${port.target ? 'connected' : ''}`, dataset: { port: 'next' }, title: 'Drag to connect' }));
    }
    return el;
  }

  summary(node) {
    const p = node.params || {};
    const lines = [];
    const clip = (text, n = 90) => {
      const t = String(text || '').replace(/\s+/g, ' ').trim();
      return t.length > n ? `${t.slice(0, n)}…` : t;
    };
    const kbNames = (ids) => (ids || []).map((id) => this.editor.kbName(id)).join(', ') || 'no knowledge base';
    switch (node.type) {
      case 'begin': {
        const inputs = Object.keys(p.inputs || {});
        lines.push(inputs.length ? `Inputs: ${inputs.join(', ')}` : 'Every run starts here');
        break;
      }
      case 'retrieval': {
        lines.push(kbNames(p.kb_ids));
        const parts = [p.mode === 'metadata' ? 'Metadata only' : clip(p.query ?? '{sys.query}', 34)];
        const filters = (p.filters || []).length;
        if (filters) parts.push(`${filters} filter${filters > 1 ? 's' : ''}`);
        if (p.sort_by) parts.push(`${p.sort_by} ${(p.sort_order || 'desc') === 'desc' ? '↓' : '↑'}`);
        parts.push(`top ${p.top_k ?? 5}`);
        lines.push(parts.join(' · '));
        break;
      }
      case 'llm':
        lines.push(p.system_prompt ? clip(p.system_prompt) : 'No system prompt');
        if (p.model || p.provider) lines.push(p.model || p.provider);
        break;
      case 'agent':
        lines.push(`Tools: ${(p.tools || []).map((t) => (t.metadata_fields?.length ? `${t.name} (+ metadata)` : t.name)).join(', ') || 'none'}`);
        if (p.system_prompt) lines.push(clip(p.system_prompt, 60));
        break;
      case 'categorize':
        if (p.input && p.input !== '{sys.query}') lines.push(`Classifies ${clip(p.input, 40)}`);
        break;
      case 'message':
        lines.push(p.text ? clip(p.text) : 'No text');
        break;
      default:
    }
    if (!lines.length) return null;
    return h('div', { class: 'node-body' }, lines.map((line) => h('div', { class: 'node-line' }, line)));
  }

  nodeEl(id) {
    return this.nodeLayer.querySelector(`.node[data-id="${CSS.escape(id)}"]`);
  }

  // Selection changes only toggle classes (a full render would replace a node being dragged).
  updateSelection() {
    const sel = this.editor.selection;
    for (const el of this.nodeLayer.children) {
      el.classList.toggle('selected', sel?.type === 'node' && sel.id === el.dataset.id);
    }
    this.drawEdges();
  }

  portPoint(nodeEl, selector) {
    const port = nodeEl.querySelector(selector);
    if (!port) return null;
    let x = port.offsetWidth / 2, y = port.offsetHeight / 2;
    for (let cur = port; cur && cur !== nodeEl; cur = cur.offsetParent) {
      x += cur.offsetLeft;
      y += cur.offsetTop;
    }
    return { x: parseFloat(nodeEl.style.left) + nodeEl.clientLeft + x, y: parseFloat(nodeEl.style.top) + nodeEl.clientTop + y };
  }

  drawEdges() {
    const { dsl, selection, runState } = this.editor;
    const els = {};
    for (const el of this.nodeLayer.children) els[el.dataset.id] = el;
    const groups = [];
    this.edgeButton.hidden = true;
    for (const e of edges(dsl)) {
      if (!els[e.from] || !els[e.to]) continue;
      const a = this.portPoint(els[e.from], `.port.out[data-port="${e.port}"]`);
      const b = this.portPoint(els[e.to], '.port.in');
      if (!a || !b) continue;
      const end = { x: b.x - 6, y: b.y };
      const selected = selection?.type === 'edge' && selection.node === e.from && selection.port === e.port;
      const ran = runState[e.from]?.status === 'done' && runState[e.to];
      const g = svg('g', { class: `edge${selected ? ' selected' : ''}${ran ? ' ran' : ''}`, 'data-from': e.from, 'data-port': e.port });
      const d = curve(a, end);
      g.append(svg('path', { d, class: 'edge-hit' }), svg('path', { d, class: 'edge-line' }),
        svg('path', { d: `M${end.x - 7},${end.y - 4.5} L${end.x + 1},${end.y} L${end.x - 7},${end.y + 4.5} Z`, class: 'edge-arrow' }));
      groups.push(g);
      if (selected) {
        const mid = midpoint(a, end);
        this.edgeButton.style.left = `${mid.x}px`;
        this.edgeButton.style.top = `${mid.y}px`;
        this.edgeButton.hidden = false;
      }
    }
    const g = this.gesture;
    if (g?.kind === 'connect' && g.point) {
      groups.push(svg('path', { d: curve(g.start, g.point), class: 'edge-line connecting' }));
    }
    this.edgeLayer.replaceChildren(...groups);
  }

  // --- View ----------------------------------------------------------------------------

  applyTransform() {
    this.viewport.style.transform = `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`;
    const size = 22 * this.zoom;
    this.root.style.backgroundSize = `${size}px ${size}px`;
    this.root.style.backgroundPosition = `${this.pan.x}px ${this.pan.y}px`;
  }

  toWorld(clientX, clientY) {
    const r = this.root.getBoundingClientRect();
    return { x: (clientX - r.left - this.pan.x) / this.zoom, y: (clientY - r.top - this.pan.y) / this.zoom };
  }

  // The world point at the middle of the visible canvas.
  center() {
    const r = this.root.getBoundingClientRect();
    return this.toWorld(r.left + r.width / 2, r.top + r.height / 2);
  }

  zoomBy(factor, clientX, clientY) {
    const r = this.root.getBoundingClientRect();
    const cx = clientX ?? r.left + r.width / 2;
    const cy = clientY ?? r.top + r.height / 2;
    const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
    const sx = cx - r.left, sy = cy - r.top;
    this.pan.x = sx - ((sx - this.pan.x) * zoom) / this.zoom;
    this.pan.y = sy - ((sy - this.pan.y) * zoom) / this.zoom;
    this.zoom = zoom;
    this.applyTransform();
  }

  fit() {
    const els = [...this.nodeLayer.children];
    if (!els.length || !this.root.clientWidth) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const el of els) {
      const x = parseFloat(el.style.left), y = parseFloat(el.style.top);
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + el.offsetWidth); maxY = Math.max(maxY, y + el.offsetHeight);
    }
    const pad = 60;
    const W = this.root.clientWidth, H = this.root.clientHeight;
    // Show everything, but not smaller than readable: a big workflow starts at its left (begin) side.
    const zoom = Math.min(1.1, Math.max(FIT_MIN_ZOOM, Math.min((W - pad * 2) / (maxX - minX), (H - pad * 2) / (maxY - minY))));
    const w = (maxX - minX) * zoom, h = (maxY - minY) * zoom;
    this.zoom = zoom;
    this.pan.x = w <= W - pad * 2 ? (W - w) / 2 - minX * zoom : pad - minX * zoom;
    this.pan.y = h <= H - pad * 2 ? (H - h) / 2 - minY * zoom : pad - minY * zoom;
    this.applyTransform();
  }

  // Scroll so a node is visible.
  reveal(id) {
    const el = this.nodeEl(id);
    if (!el) return;
    const x = parseFloat(el.style.left) * this.zoom + this.pan.x;
    const y = parseFloat(el.style.top) * this.zoom + this.pan.y;
    const W = this.root.clientWidth, H = this.root.clientHeight;
    if (x < 20 || y < 20 || x + el.offsetWidth * this.zoom > W - 20 || y + el.offsetHeight * this.zoom > H - 20) {
      this.pan.x += W / 2 - (x + (el.offsetWidth * this.zoom) / 2);
      this.pan.y += H / 2 - (y + (el.offsetHeight * this.zoom) / 2);
      this.applyTransform();
    }
  }

  // Sizes of the rendered nodes, for auto layout.
  sizes() {
    const sizes = {};
    for (const el of this.nodeLayer.children) sizes[el.dataset.id] = { w: el.offsetWidth, h: el.offsetHeight };
    return sizes;
  }

  // --- Interaction ---------------------------------------------------------------------

  bind() {
    const root = this.root;
    root.addEventListener('pointerdown', (e) => this.onDown(e));
    window.addEventListener('pointermove', this.onMove = (e) => this.move(e));
    window.addEventListener('pointerup', this.onUp = (e) => this.up(e));
    window.addEventListener('pointercancel', this.onUp);
    window.addEventListener('blur', this.onBlur = () => { // e.g. alt-tab in the middle of a drag
      if (this.gesture) { this.gesture = null; this.render(); }
    });
    root.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey || !e.shiftKey) {
        this.zoomBy(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX, e.clientY);
      } else {
        this.pan.x -= e.deltaY;
        this.applyTransform();
      }
    }, { passive: false });
    root.addEventListener('dragover', (e) => {
      if (e.dataTransfer.types.includes('application/x-node-type')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }
    });
    root.addEventListener('drop', (e) => {
      const type = e.dataTransfer.getData('application/x-node-type');
      if (!type) return;
      e.preventDefault();
      const p = this.toWorld(e.clientX, e.clientY);
      this.editor.addNode(type, { x: p.x - 120, y: p.y - 24 });
    });
  }

  destroy() {
    window.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    window.removeEventListener('pointercancel', this.onUp);
    window.removeEventListener('blur', this.onBlur);
  }

  onDown(e) {
    if (e.button !== 0 && e.button !== 1) return;
    if (e.target.closest('.canvas-controls, .edge-delete')) return;
    // Taking the focus ends an edit in the panel, which may redraw the nodes (a node rename):
    // look at what is under the pointer after that, not at the element that got the event.
    this.root.focus({ preventScroll: true });
    const under = document.elementFromPoint(e.clientX, e.clientY) || e.target;
    const port = e.button === 0 && under.closest('.port.out');
    const nodeEl = under.closest('.node');
    if (port && nodeEl) {
      e.preventDefault();
      const start = this.portPoint(nodeEl, `.port.out[data-port="${port.dataset.port}"]`);
      this.gesture = { kind: 'connect', from: nodeEl.dataset.id, port: port.dataset.port, start, point: null, x0: e.clientX, y0: e.clientY };
      return;
    }
    if (nodeEl && e.button === 0) {
      e.preventDefault();
      const id = nodeEl.dataset.id;
      this.editor.select({ type: 'node', id });
      const pos = this.editor.position(id);
      this.gesture = { kind: 'drag', id, x0: e.clientX, y0: e.clientY, start: { ...pos }, moved: false };
      return;
    }
    const edge = e.button === 0 && under.closest('.edge');
    if (edge) {
      this.editor.select({ type: 'edge', node: edge.dataset.from, port: edge.dataset.port });
      return;
    }
    e.preventDefault();
    this.gesture = { kind: 'pan', x0: e.clientX, y0: e.clientY, start: { ...this.pan }, moved: false };
  }

  move(e) {
    const g = this.gesture;
    if (!g) return;
    if (e.buttons === 0) return this.up(e); // released outside the window
    const dx = e.clientX - g.x0, dy = e.clientY - g.y0;
    if (g.kind === 'pan') {
      if (Math.abs(dx) + Math.abs(dy) > 3) g.moved = true;
      this.pan.x = g.start.x + dx;
      this.pan.y = g.start.y + dy;
      this.applyTransform();
    } else if (g.kind === 'drag') {
      if (!g.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      g.moved = true;
      const el = this.nodeEl(g.id);
      if (!el) return;
      g.pos = { x: Math.round(g.start.x + dx / this.zoom), y: Math.round(g.start.y + dy / this.zoom) };
      el.style.left = `${g.pos.x}px`;
      el.style.top = `${g.pos.y}px`;
      el.classList.add('dragging');
      this.drawEdges();
    } else if (g.kind === 'connect') {
      g.point = this.toWorld(e.clientX, e.clientY);
      for (const el of this.nodeLayer.querySelectorAll('.drop-target')) el.classList.remove('drop-target');
      const target = this.dropTarget(e, g);
      if (target) target.classList.add('drop-target');
      this.drawEdges();
    }
  }

  // The node a connection being dragged from g.from would go to (not itself, not begin).
  dropTarget(e, g) {
    const el = document.elementFromPoint(e.clientX, e.clientY)?.closest('.node');
    if (!el || el.dataset.id === g.from || this.editor.dsl.nodes[el.dataset.id]?.type === 'begin') return null;
    return el;
  }

  up(e) {
    const g = this.gesture;
    if (!g) return;
    this.gesture = null;
    if (g.kind === 'pan') {
      if (!g.moved) this.editor.select(null);
    } else if (g.kind === 'drag') {
      this.nodeEl(g.id)?.classList.remove('dragging');
      if (g.moved && g.pos) this.editor.moveNode(g.id, g.pos);
    } else if (g.kind === 'connect') {
      for (const el of this.nodeLayer.querySelectorAll('.drop-target')) el.classList.remove('drop-target');
      const target = this.dropTarget(e, g);
      const moved = Math.abs(e.clientX - g.x0) + Math.abs(e.clientY - g.y0) > 6;
      if (target) {
        this.editor.connect(g.from, g.port, target.dataset.id);
      } else if (moved && e.target.closest('.canvas') === this.root && !e.target.closest('.node')) {
        const p = this.toWorld(e.clientX, e.clientY);
        this.editor.quickAdd({ x: e.clientX, y: e.clientY }, { x: p.x + 10, y: p.y - 24 }, { node: g.from, port: g.port });
      } else if (!moved) {
        this.editor.select({ type: 'node', id: g.from });
      }
      this.drawEdges();
    }
  }
}

function curve(a, b) {
  const back = b.x < a.x + 30;
  const dx = back ? Math.max(90, (a.x - b.x) * 0.4) : Math.max(40, (b.x - a.x) * 0.5);
  return `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`;
}

function midpoint(a, b) {
  const back = b.x < a.x + 30;
  const dx = back ? Math.max(90, (a.x - b.x) * 0.4) : Math.max(40, (b.x - a.x) * 0.5);
  // Cubic bezier at t = 0.5: (p0 + 3 p1 + 3 p2 + p3) / 8
  return { x: (a.x + 3 * (a.x + dx) + 3 * (b.x - dx) + b.x) / 8, y: (a.y + 3 * a.y + 3 * b.y + b.y) / 8 };
}

// The workflow editor page: owns the working copy of the workflow, undo/redo, saving and validation.
// The canvas, the settings forms and the test panel read from it and change it through change().

import { api, config } from './api.js';
import { Canvas } from './canvas.js';
import { outputList, renderEdgePanel, renderNodePanel, renderWorkflowPanel } from './form.js';
import { autoLayout, clearTargets, newNodeId, outPorts, removeNode, renameNode, setTarget } from './graph.js';
import { RunLog } from './runlog.js';
import { TestPanel } from './testrun.js';
import { button, clone, closePopover, confirmDialog, downloadJson, h, icon, iconButton, modal, popover, slug, toast } from './ui.js';

const PALETTE = ['retrieval', 'llm', 'agent', 'categorize', 'switch', 'message'];

export class EditorPage {
  constructor(root, app, id) {
    this.root = root;
    this.app = app;
    this.types = app.types;
    this.sysVars = app.sysVars;
    this.modelDefaults = app.modelDefaults;
    this.kbs = app.kbs;
    this.apiRoot = config.apiRoot;
    this.selection = null;
    this.errors = [];
    this.errorsByNode = {};
    this.generalErrors = [];
    this.runState = {};
    this.undoStack = [];
    this.redoStack = [];
    this.dirty = false;
    this.saving = false;
    this.openGroups = new Set();
    this.tab = 'settings';
    this.problemsOpen = false;
    this.validateSeq = 0;
    this.onKey = (e) => this.handleKey(e);
    root.replaceChildren(h('div', { class: 'boot' }, 'Loading…'));
    this.load(id);
  }

  async load(id) {
    try {
      const [workflow, kbs] = await Promise.all([api.get(`workflows/${id}`), api.get('kbs')]);
      this.workflow = workflow;
      this.kbs = this.app.kbs = kbs;
    } catch (err) {
      if (this.destroyed) return;
      this.root.replaceChildren(h('div', { class: 'boot error' },
        h('h1', null, err.status === 404 ? 'Workflow not found' : 'Could not open the workflow'),
        h('p', null, err.message), h('a', { href: '#/', class: 'btn' }, 'All workflows')));
      return;
    }
    if (this.destroyed) return;
    this.name = this.workflow.name;
    this.description = this.workflow.description;
    this.dsl = normalize(clone(this.workflow.dsl));
    this.build();
    this.canvas.render();
    if (this.placeMissing()) this.canvas.render(); // e.g. made from a template or with the API
    this.savedSnapshot = this.snapshot();
    this.updateStatus();
    this.renderSidebar();
    requestAnimationFrame(() => this.canvas.fit());
    this.validateSoon(0);
    document.addEventListener('keydown', this.onKey);
  }

  destroy() {
    this.destroyed = true;
    this.canvas?.destroy();
    this.testPanel?.controller?.abort();
    clearTimeout(this.validateTimer);
    document.removeEventListener('keydown', this.onKey);
    closePopover();
  }

  // --- Layout ---------------------------------------------------------------------------

  build() {
    this.canvas = new Canvas(this);
    this.testPanel = new TestPanel(this);
    this.runLog = new RunLog(this);
    this.titleEl = h('button', { type: 'button', class: 'wf-title', title: 'Workflow settings', onClick: () => this.select(null) }, this.name);
    this.status = h('span', { class: 'save-status' });
    this.undoBtn = iconButton('undo', 'Undo (Ctrl+Z)', () => this.undo());
    this.redoBtn = iconButton('redo', 'Redo (Ctrl+Shift+Z)', () => this.redo());
    this.saveBtn = button('Save', { icon: 'save', kind: 'primary', title: 'Save (Ctrl+S)', onClick: () => this.save() });
    this.tabButtons = {
      settings: h('button', { type: 'button', class: 'tab', onClick: () => this.showTab('settings') }, icon('settings'), 'Settings'),
      test: h('button', { type: 'button', class: 'tab', onClick: () => this.showTab('test') }, icon('play'), 'Test run'),
    };
    this.sidebarBody = h('div', { class: 'sidebar-body' });
    this.problemsEl = h('div', { class: 'problems-bar', hidden: true });

    const palette = h('aside', { class: 'palette', 'aria-label': 'Node types' },
      h('div', { class: 'palette-title' }, 'Add a node'),
      PALETTE.filter((t) => this.types[t]).map((type) => {
        const info = this.types[type];
        const item = h('button', {
          type: 'button', class: `palette-item t-${type}`, draggable: 'true', title: `${info.description}\nClick to add after the selected node, or drag onto the canvas.`,
          onClick: () => this.addNode(type),
        }, h('span', { class: 'node-icon' }, icon(type)),
        h('span', { class: 'palette-text' }, h('span', { class: 'palette-name' }, info.title), h('span', { class: 'palette-desc' }, info.description)));
        item.addEventListener('dragstart', (e) => {
          e.dataTransfer.setData('application/x-node-type', type);
          e.dataTransfer.effectAllowed = 'copy';
        });
        return item;
      }),
      h('div', { class: 'shortcuts' },
        h('div', { class: 'palette-title' }, 'Shortcuts'),
        shortcut('Ctrl S', 'Save'), shortcut('Ctrl Z', 'Undo'), shortcut('Ctrl ⇧ Z', 'Redo'),
        shortcut('Ctrl D', 'Duplicate'), shortcut('Del', 'Delete'), shortcut('Wheel', 'Zoom'), shortcut('Drag', 'Pan')));

    this.root.replaceChildren(h('div', { class: 'editor' },
      h('header', { class: 'topbar' },
        h('a', { class: 'icon-btn', href: '#/', title: 'All workflows', 'aria-label': 'All workflows' }, icon('back')),
        this.titleEl, this.status,
        h('span', { class: 'spacer' }),
        this.undoBtn, this.redoBtn,
        button('Test', { icon: 'play', onClick: () => this.showTab('test') }),
        this.saveBtn),
      h('div', { class: 'editor-main' },
        palette,
        h('div', { class: 'canvas-wrap' }, h('div', { class: 'canvas-area' }, this.canvas.root, this.problemsEl), this.runLog.el),
        h('aside', { class: 'sidebar' },
          h('div', { class: 'tabs', role: 'tablist' }, this.tabButtons.settings, this.tabButtons.test),
          this.sidebarBody, this.testPanel.el))));
    this.showTab('settings');
  }

  showTab(tab) {
    this.tab = tab;
    for (const [name, el] of Object.entries(this.tabButtons)) el.classList.toggle('active', name === tab);
    this.sidebarBody.hidden = tab !== 'settings';
    this.testPanel.el.hidden = tab !== 'test';
    if (tab === 'test') {
      this.testPanel.renderInputs();
      this.testPanel.input.focus();
    }
  }

  // A field that commits on change (the node id) commits when it loses focus: do that now.
  commitEdits() {
    const active = document.activeElement;
    if (active && active !== document.body && this.root.contains(active) && active.matches('input, textarea, select')) active.blur();
  }

  renderSidebar() {
    if (!this.sidebarBody) return;
    closePopover(); // a variable menu would insert into a field that is about to be replaced
    const active = document.activeElement;
    const focusId = this.sidebarBody.contains(active) && active.id ? active.id : null;
    const caret = focusId && typeof active.selectionStart === 'number' ? [active.selectionStart, active.selectionEnd] : null;
    const sel = this.selection;
    const key = sel ? JSON.stringify(sel) : 'workflow';
    const scroll = this.sidebarBody.scrollTop;
    let panel;
    if (sel?.type === 'node' && this.dsl.nodes[sel.id]) panel = renderNodePanel(this, sel.id);
    else if (sel?.type === 'edge' && this.dsl.nodes[sel.node]) panel = renderEdgePanel(this, sel);
    else panel = renderWorkflowPanel(this);
    this.sidebarBody.replaceChildren(panel);
    this.sidebarBody.scrollTop = key === this.panelKey ? scroll : 0;
    this.panelKey = key;
    const again = focusId && document.getElementById(focusId);
    if (again && this.sidebarBody.contains(again)) {
      again.focus({ preventScroll: true });
      if (caret && typeof again.selectionStart === 'number') again.setSelectionRange(...caret);
    }
  }

  updateStatus() {
    this.titleEl.textContent = this.name || 'Untitled';
    this.status.textContent = this.saving ? 'Saving…' : this.dirty ? 'Unsaved changes' : 'All changes saved';
    this.status.classList.toggle('dirty', this.dirty);
    this.saveBtn.disabled = this.saving || !this.dirty;
    this.undoBtn.disabled = !this.undoStack.length;
    this.redoBtn.disabled = !this.redoStack.length;
  }

  // --- Changes and undo -------------------------------------------------------------------

  snapshot() {
    return JSON.stringify({ dsl: this.dsl, name: this.name, description: this.description });
  }

  // Every edit goes through here. Edits with the same key in a row (typing) are one undo step.
  // source: 'form' (the panel changed itself: don't redraw it), 'layout' (positions only), or other.
  change(key, fn, { source = 'canvas' } = {}) {
    const now = Date.now();
    const before = this.snapshot();
    const coalesce = key && key === this.lastKey && now - this.lastTime < 1500;
    fn();
    if (this.snapshot() === before) return; // nothing changed: no undo step, keep redo
    if (!coalesce) {
      this.undoStack.push(before);
      if (this.undoStack.length > 200) this.undoStack.shift();
      this.redoStack = [];
    }
    this.lastKey = key;
    this.lastTime = now;
    this.afterChange(source);
  }

  afterChange(source) {
    this.dirty = this.snapshot() !== this.savedSnapshot;
    this.updateStatus();
    this.canvas.scheduleRender();
    if (source !== 'layout') this.validateSoon();
    if (source !== 'form' && source !== 'layout') this.renderSidebar();
    if (source === 'form' && this.selection?.type === 'node' && this.dsl.nodes[this.selection.id]?.type === 'begin') {
      this.sidebarBody.querySelector('.outputs-slot')?.replaceChildren(outputList(this, this.selection.id));
    }
  }

  restore(snapshot) {
    const s = JSON.parse(snapshot);
    this.dsl = s.dsl;
    this.name = s.name;
    this.description = s.description;
    this.lastKey = null;
    if (this.selection?.type === 'node' && !this.dsl.nodes[this.selection.id]) this.selection = null;
    if (this.selection?.type === 'edge' && !this.dsl.nodes[this.selection.node]) this.selection = null;
    this.canvas.render();
    this.afterChange('undo');
  }

  undo() {
    if (!this.undoStack.length) return;
    this.redoStack.push(this.snapshot());
    this.restore(this.undoStack.pop());
  }

  redo() {
    if (!this.redoStack.length) return;
    this.undoStack.push(this.snapshot());
    this.restore(this.redoStack.pop());
  }

  // --- Selection ----------------------------------------------------------------------------

  select(selection, { keepTab = false } = {}) {
    const same = JSON.stringify(selection) === JSON.stringify(this.selection);
    this.selection = selection;
    this.canvas.updateSelection();
    this.runLog?.highlight();
    if (!keepTab) this.showTab('settings');
    if (selection?.type === 'node' && keepTab) this.canvas.reveal(selection.id);
    if (!same) this.renderSidebar();
  }

  // --- Graph edits ----------------------------------------------------------------------------

  position(id) {
    return this.dsl.layout?.[id] || { x: 0, y: 0 };
  }

  kbName(id) {
    return this.kbs.find((kb) => kb.id === id)?.name || `#${id}`;
  }

  // The metadata fields of these knowledge bases ({name, type, count, examples}), for field
  // suggestions. Asked once per knowledge base while the editor is open.
  async metadataFields(kbIds) {
    this.fieldCache = this.fieldCache || {};
    const ids = [...new Set(kbIds)].filter((id) => this.kbs.some((kb) => kb.id === id));
    const lists = await Promise.all(ids.map((id) => {
      this.fieldCache[id] = this.fieldCache[id] || api.get(`kbs/${id}/metadata-fields`).catch(() => {
        delete this.fieldCache[id]; // try again next time
        return [];
      });
      return this.fieldCache[id];
    }));
    const byName = new Map();
    for (const list of lists) {
      for (const m of list) {
        const seen = byName.get(m.name);
        if (!seen) byName.set(m.name, { ...m, examples: [...m.examples] });
        else {
          seen.count += m.count;
          for (const x of m.examples) if (seen.examples.length < 3 && !seen.examples.includes(x)) seen.examples.push(x);
        }
      }
    }
    return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  moveNode(id, pos) {
    if (!this.dsl.nodes[id]) return;
    this.change(null, () => {
      this.dsl.layout = this.dsl.layout || {};
      this.dsl.layout[id] = pos;
    }, { source: 'layout' });
  }

  connect(from, port, to) {
    if (to === from || !this.dsl.nodes[from] || (to && !this.dsl.nodes[to])) return;
    this.change(null, () => setTarget(this.dsl.nodes[from], port, to));
  }

  addNode(type, pos = null, from = null) {
    const info = this.types[type];
    if (!info || type === 'begin') return;
    const node = { type, params: clone(info.starter || {}) };
    const firstKb = this.kbs[0]?.id;
    if (firstKb !== undefined) {
      if (type === 'retrieval' && !node.params.kb_ids?.length) node.params.kb_ids = [firstKb];
      for (const tool of node.params.tools || []) if (!tool.kb_ids?.length) tool.kb_ids = [firstKb];
    }

    // Clicked in the palette: add it after the selected node, on its first free output,
    // or between the node and its next one.
    let source = from;
    let insertBefore = null;
    if (!source && !pos && this.selection?.type === 'node') {
      const selected = this.dsl.nodes[this.selection.id];
      const ports = selected ? outPorts(selected) : [];
      const free = ports.find((p) => !p.target);
      if (free) source = { node: this.selection.id, port: free.key };
      else if (ports.length === 1) {
        source = { node: this.selection.id, port: ports[0].key };
        insertBefore = ports[0].target;
      }
    }
    const id = newNodeId(this.dsl, type);
    if (source) this.smartDefaults(node, source.node);
    if (insertBefore) setTarget(node, outPorts(node)[0]?.key || 'next', insertBefore);

    if (!pos) {
      if (source) {
        const src = this.position(source.node);
        const index = Math.max(0, outPorts(this.dsl.nodes[source.node]).findIndex((p) => p.key === source.port));
        pos = { x: src.x + 300, y: src.y + index * 80 };
      } else {
        const c = this.canvas.center();
        pos = { x: Math.round(c.x - 120), y: Math.round(c.y - 40) };
      }
    }
    this.change(null, () => {
      this.dsl.layout = this.dsl.layout || {};
      if (insertBefore) {
        // Make room: everything right of the source moves one column right.
        const srcX = this.position(source.node).x;
        for (const [nodeId, p] of Object.entries(this.dsl.layout)) if (nodeId !== source.node && p.x > srcX + 20) p.x += 300;
      } else pos = this.freeSpot(pos);
      this.dsl.nodes[id] = node;
      this.dsl.layout[id] = pos;
      if (source) setTarget(this.dsl.nodes[source.node], source.port, id);
    });
    this.select({ type: 'node', id });
    this.canvas.root.focus({ preventScroll: true }); // so Delete / Ctrl+D act on it
    requestAnimationFrame(() => this.canvas.reveal(id));
  }

  // Useful first settings for a node added after another one.
  smartDefaults(node, sourceId) {
    const source = this.dsl.nodes[sourceId];
    if (!source) return;
    if (node.type === 'message' && ['llm', 'agent'].includes(source.type)) node.params.text = `{${sourceId}@content}`;
    if (node.type === 'llm' && source.type === 'retrieval') {
      node.params.system_prompt = 'Answer the user\'s question using only the information below. If it doesn\'t contain the answer, '
        + 'say you don\'t know. Cite the numbers of the items you use, like [1]. Answer in the user\'s language.'
        + `\n\n# Information\n{${sourceId}@formalized_content}`;
    }
    if (node.type === 'retrieval' && source.type === 'llm') node.params.query = `{${sourceId}@content}`;
  }

  // The nearest spot at or below pos where the new node doesn't cover another one.
  freeSpot(pos) {
    const sizes = this.canvas.sizes();
    const taken = Object.entries(this.dsl.layout || {}).map(([id, p]) => ({ ...p, h: sizes[id]?.h || 100 }));
    let { x, y } = pos;
    const covers = () => taken.find((p) => Math.abs(p.x - x) < 250 && y < p.y + p.h + 16 && p.y < y + 100);
    for (let i = 0, hit; i < 50 && (hit = covers()); i++) y = hit.y + hit.h + 24;
    return { x: Math.round(x), y: Math.round(y) };
  }

  quickAdd(screen, world, from) {
    const content = h('div', { class: 'quick-add' },
      h('div', { class: 'var-group-title' }, 'Add and connect'),
      PALETTE.filter((t) => this.types[t]).map((type) => h('button', {
        type: 'button', class: `quick-item t-${type}`,
        onClick: () => { closePopover(); this.addNode(type, world, from); },
      }, h('span', { class: 'node-icon' }, icon(type)), this.types[type].title)));
    popover(screen, content, { className: 'quick-popover' });
  }

  deleteNode(id) {
    const node = this.dsl.nodes[id];
    if (!node) return;
    if (node.type === 'begin') {
      toast('Every workflow needs its begin node', 'error');
      return;
    }
    this.change(null, () => removeNode(this.dsl, id));
    this.select(null);
  }

  deleteSelection() {
    const sel = this.selection;
    if (sel?.type === 'node') this.deleteNode(sel.id);
    else if (sel?.type === 'edge') {
      this.change(null, () => setTarget(this.dsl.nodes[sel.node], sel.port, null));
      this.select(null);
    }
  }

  duplicateNode(id) {
    const source = this.dsl.nodes[id];
    if (!source) return;
    if (source.type === 'begin') {
      toast('A workflow has exactly one begin node', 'error');
      return;
    }
    const copy = clone(source);
    clearTargets(copy);
    const newId = newNodeId(this.dsl, id.replace(/_\d+$/, ''));
    const pos = this.freeSpot({ x: this.position(id).x + 40, y: this.position(id).y + 60 });
    this.change(null, () => {
      this.dsl.nodes[newId] = copy;
      this.dsl.layout[newId] = pos;
    });
    this.select({ type: 'node', id: newId });
  }

  renameNode(oldId, newId) {
    this.change(null, () => renameNode(this.dsl, oldId, newId), { source: 'layout' });
    if (this.selection?.type === 'node' && this.selection.id === oldId) this.selection = { type: 'node', id: newId };
    this.canvas.render();
    // The rename runs when the id field loses focus, often in the middle of a click elsewhere
    // in the panel: redraw after it, and renderSidebar puts the focus back on that field.
    setTimeout(() => { if (!this.destroyed) this.renderSidebar(); });
    this.validateSoon();
  }

  // Positions for nodes without one (made with the API, or in the JSON editor), next to a node before them.
  placeMissing() {
    this.dsl.layout = this.dsl.layout || {};
    const layout = this.dsl.layout;
    const missing = Object.keys(this.dsl.nodes).filter((id) => !layout[id]);
    if (!missing.length) return false;
    if (missing.length === Object.keys(this.dsl.nodes).length) {
      this.dsl.layout = autoLayout(this.dsl, this.canvas.sizes());
      return true;
    }
    for (const id of missing) {
      const parent = Object.keys(this.dsl.nodes).find((p) => layout[p] && outPorts(this.dsl.nodes[p]).some((port) => port.target === id));
      const below = Math.max(...Object.values(layout).map((p) => p.y)) + 160;
      layout[id] = this.freeSpot(parent ? { x: layout[parent].x + 300, y: layout[parent].y } : { x: 0, y: below });
    }
    return true;
  }

  autoArrange() {
    this.change(null, () => { this.dsl.layout = autoLayout(this.dsl, this.canvas.sizes()); }, { source: 'layout' });
    this.canvas.render();
    requestAnimationFrame(() => this.canvas.fit());
  }

  // The workflow as sent to the server: positions only for nodes that exist.
  payloadDsl() {
    const dsl = clone(this.dsl);
    const layout = {};
    for (const id of Object.keys(dsl.nodes)) {
      const p = this.dsl.layout?.[id];
      if (p) layout[id] = { x: Math.round(p.x), y: Math.round(p.y) };
    }
    dsl.layout = layout;
    return dsl;
  }

  // --- Validation and run state ----------------------------------------------------------------

  validateSoon(delay = 400) {
    clearTimeout(this.validateTimer);
    this.validateTimer = setTimeout(async () => {
      const seq = ++this.validateSeq;
      try {
        const result = await api.post('workflows/validate', { dsl: this.payloadDsl() });
        if (seq === this.validateSeq && !this.destroyed) this.setErrors(result.errors);
      } catch { /* offline: the next save will tell */ }
    }, delay);
  }

  setErrors(errors) {
    this.errors = errors || [];
    this.errorsByNode = {};
    this.generalErrors = [];
    for (const e of this.errors) {
      if (e.node && this.dsl.nodes[e.node]) (this.errorsByNode[e.node] = this.errorsByNode[e.node] || []).push(e.message);
      else this.generalErrors.push(e.node ? `${e.node}: ${e.message}` : e.message);
    }
    this.canvas.scheduleRender();
    this.renderProblems();
    this.updatePanelProblems();
  }

  // Refresh the problems box in the open panel without redrawing its fields (the user may be typing).
  updatePanelProblems() {
    const slot = this.sidebarBody?.querySelector('.problems-slot');
    if (!slot) return;
    const sel = this.selection;
    const messages = sel?.type === 'node' ? this.errorsByNode[sel.id] : sel ? null : this.generalErrors;
    slot.replaceChildren(...(messages || []).map((m) => h('div', { class: 'problem' }, icon('alert'), h('span', null, m))));
  }

  renderProblems() {
    const count = this.errors.length;
    this.problemsEl.hidden = !count;
    if (!count) return;
    this.problemsEl.replaceChildren(
      h('button', { type: 'button', class: 'problems-toggle', onClick: () => { this.problemsOpen = !this.problemsOpen; this.renderProblems(); } },
        icon('alert'), `${count} problem${count > 1 ? 's' : ''} to fix before saving`, icon(this.problemsOpen ? 'chevronDown' : 'chevronUp')),
      this.problemsOpen ? h('div', { class: 'problems-list' }, this.errors.map((e) => h('button', {
        type: 'button', class: 'problem-item',
        onClick: () => this.select(e.node && this.dsl.nodes[e.node] ? { type: 'node', id: e.node } : null, { keepTab: false }),
      }, e.node ? h('code', null, e.node) : null, h('span', null, e.message)))) : '');
  }

  setRunState(state) {
    this.runState = state;
    this.canvas.scheduleRender();
  }

  // --- Saving and files --------------------------------------------------------------------------

  async save({ force = false } = {}) {
    this.commitEdits();
    if (this.saving || (!this.dirty && !force)) return;
    if (!this.name.trim()) {
      toast('Give the workflow a name first', 'error');
      this.select(null);
      return;
    }
    this.saving = true;
    this.updateStatus();
    const snapshot = this.snapshot();
    try {
      this.workflow = await api.put(`workflows/${this.workflow.id}`, {
        name: this.name.trim(),
        description: this.description,
        dsl: this.payloadDsl(),
        base_updated_at: force ? null : this.workflow.updated_at,
      });
      if (this.destroyed) return;
      this.savedSnapshot = snapshot;
      this.dirty = this.snapshot() !== this.savedSnapshot;
      toast('Saved. Apps using this workflow get the new version now.');
      if (!this.selection) this.renderSidebar();
    } catch (err) {
      if (err.status === 409 && /someone else/.test(err.message)) {
        this.saving = false;
        const overwrite = await confirmDialog('Save over newer changes?',
          'Someone saved this workflow after you opened it. Saving now replaces their changes with yours.',
          { confirm: 'Save anyway', danger: true });
        if (overwrite) await this.save({ force: true });
      } else if (err.errors?.length) {
        this.setErrors(err.errors);
        this.problemsOpen = true;
        this.renderProblems();
        toast(`Not saved: fix the ${err.errors.length > 1 ? `${err.errors.length} problems` : 'problem'} first`, 'error');
      } else {
        toast(`Not saved: ${err.message}`, 'error', 6000);
      }
    } finally {
      this.saving = false;
      this.updateStatus();
    }
  }

  exportJson() {
    downloadJson(`${slug(this.name)}.json`, { name: this.name, description: this.description, dsl: this.payloadDsl() });
  }

  editJson() {
    const text = h('textarea', { class: 'prompt big code', spellcheck: 'false', value: JSON.stringify(this.payloadDsl(), null, 2) });
    const error = h('div', { class: 'form-error' });
    modal({
      title: 'Edit the workflow as JSON',
      wide: true,
      body: h('div', { class: 'big-editor' }, text, error,
        h('div', { class: 'help' }, 'The same format the API takes as "dsl". Applying it is one undo step; Save to keep it.')),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Apply', kind: 'primary',
          onClick: () => {
            let data;
            try {
              data = JSON.parse(text.value);
            } catch (e) {
              error.textContent = `Not valid JSON: ${e.message}`;
              return false;
            }
            if (!data || typeof data !== 'object' || !data.nodes || typeof data.nodes !== 'object' || Array.isArray(data.nodes)) {
              error.textContent = 'A workflow is an object with a "nodes" object.';
              return false;
            }
            this.change(null, () => {
              this.dsl = normalize(data);
              this.placeMissing();
            });
            this.selection = null;
            this.canvas.render();
            this.renderSidebar();
            requestAnimationFrame(() => this.canvas.fit());
            return true;
          },
        },
      ],
    });
  }

  async deleteWorkflow() {
    const ok = await confirmDialog('Delete this workflow?',
      `"${this.workflow.name}" and its conversations will be deleted. Apps that use it will get errors.`, { confirm: 'Delete', danger: true });
    if (!ok) return;
    try {
      await api.del(`workflows/${this.workflow.id}`);
      this.dirty = false;
      toast('Workflow deleted');
      location.hash = '#/';
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  // --- Keyboard ------------------------------------------------------------------------------------

  handleKey(e) {
    const mod = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();
    if (mod && key === 's') {
      e.preventDefault();
      this.save();
      return;
    }
    if (document.querySelector('.modal-backdrop')) return;
    if (e.target.closest?.('input, textarea, select, [contenteditable]')) return;
    const onCanvas = this.canvas.root.contains(document.activeElement);
    if (mod && key === 'z') {
      e.preventDefault();
      if (e.shiftKey) this.redo(); else this.undo();
    } else if (mod && key === 'y') {
      e.preventDefault();
      this.redo();
    } else if (mod && key === 'd') {
      e.preventDefault();
      if (this.selection?.type === 'node' && onCanvas) this.duplicateNode(this.selection.id);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      // Only when the canvas has the focus: after a panel redraw the focus can fall back to the page.
      if (this.selection && onCanvas) {
        e.preventDefault();
        this.deleteSelection();
      }
    } else if (e.key === 'Escape') {
      this.select(null, { keepTab: true });
    }
  }
}

// Every node gets a params object, so the forms can edit it in place.
function normalize(dsl) {
  for (const node of Object.values(dsl.nodes || {})) {
    if (!node || typeof node !== 'object') continue;
    if (!node.params || typeof node.params !== 'object') node.params = {};
    upgradeFilters(node.params);
    for (const tool of Array.isArray(node.params.tools) ? node.params.tools : []) if (tool && typeof tool === 'object') upgradeFilters(tool);
  }
  return dsl;
}

// Workflows saved before "filters" have metadata_filter: {field: value}. The server reads it as
// == filters; show it as those here, so it can be edited (saving stores the new form).
function upgradeFilters(p) {
  if (!('metadata_filter' in p)) return;
  const old = p.metadata_filter;
  if (old !== null && (typeof old !== 'object' || Array.isArray(old))) return; // the server reports it
  const filters = Object.entries(old || {}).map(([field, value]) => ({ field, op: '==', value }));
  delete p.metadata_filter;
  if (Array.isArray(p.filters)) filters.push(...p.filters);
  if (filters.length) p.filters = filters;
}

function shortcut(keys, label) {
  return h('div', { class: 'shortcut' }, h('kbd', null, keys), h('span', null, label));
}

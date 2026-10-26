// The run log: a drawer under the canvas with every node a test run went through, and for each
// one what it worked with (prompts, messages, queries, filters, settings), the agent's tool calls,
// what it produced and any error. The test panel fills it as the run's events arrive.

import { button, downloadJson, h, icon, iconButton, toast } from './ui.js';

const HEIGHT_KEY = 'studio.runLogHeight';

const LABELS = {
  sys: 'System variables', history: 'Conversation history', request_inputs: 'Inputs in the request', defaults: 'Default inputs',
  mode: 'Find by', query: 'Query', kb_ids: 'Knowledge bases', filters: 'Filters', sort_by: 'Sort by', sort_order: 'Order',
  sort_pool: 'Matches to sort', top_k: 'Results', max_distance: 'Max vector distance',
  model: 'Model', system_prompt: 'System prompt', messages: 'Messages sent to the model', tools: 'Tools given to the model',
  max_rounds: 'Max tool rounds', categories: 'Categories', cases: 'Cases checked', text: 'Text', streamed_from: 'Streamed from',
  skipped_filters: 'Filters skipped (empty value)',
};

export class RunLog {
  constructor(editor) {
    this.editor = editor;
    this.turn = null;
    this.closedByUser = false;
    this.queued = false;
    this.height = loadHeight();
    this.picker = h('select', {
      class: 'log-picker', 'aria-label': 'Which message',
      onChange: (e) => this.show(this.turns()[Number(e.target.value)]),
    });
    this.summary = h('span', { class: 'log-summary' });
    this.body = h('div', { class: 'log-body' });
    const handle = h('div', { class: 'log-resize', title: 'Drag to resize', role: 'separator', 'aria-orientation': 'horizontal' });
    handle.addEventListener('pointerdown', (e) => this.startResize(e));
    this.el = h('section', { class: 'run-log', hidden: true, 'aria-label': 'Run log' },
      handle,
      h('header', { class: 'log-head' },
        h('span', { class: 'log-title' }, icon('log'), 'Run log'),
        this.picker, this.summary,
        h('span', { class: 'spacer' }),
        button('Expand all', { kind: 'small', onClick: () => this.expandAll(true) }),
        button('Collapse all', { kind: 'small', onClick: () => this.expandAll(false) }),
        iconButton('copy', 'Copy the log as JSON', () => this.copy()),
        iconButton('download', 'Download the log as JSON', () => this.download()),
        iconButton('x', 'Close the log', () => this.hide(true))),
      this.body);
    this.applyHeight();
  }

  turns() {
    return this.editor.testPanel.turns;
  }

  get isOpen() {
    return !this.el.hidden;
  }

  show(turn) {
    this.turn = turn || null;
    this.el.hidden = false;
    this.render();
  }

  hide(byUser = false) {
    this.el.hidden = true;
    if (byUser) this.closedByUser = true; // runs don't open it again until the user does
  }

  // From the test panel: open on `turn`, or close when it is showing already.
  toggle(turn) {
    if (this.isOpen && (!turn || turn === this.turn)) this.hide(true);
    else {
      this.closedByUser = false;
      this.show(turn || this.turns().at(-1));
    }
  }

  // A run started: follow it, unless the user closed the log.
  follow(turn) {
    if (!this.closedByUser) this.show(turn);
  }

  // An event arrived for `turn`.
  update(turn) {
    if (!this.isOpen || turn !== this.turn || this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      if (this.isOpen && !this.editor.destroyed) this.render();
    });
  }

  reset() {
    this.turn = null;
    if (this.isOpen) this.render();
  }

  // The steps of the selected node stand out (called when the selection changes).
  highlight() {
    const sel = this.editor.selection;
    for (const el of this.body.querySelectorAll('.log-step')) {
      el.classList.toggle('selected', sel?.type === 'node' && sel.id === el.dataset.node);
    }
  }

  expandAll(open) {
    const turn = this.turn;
    if (!turn) return;
    turn.logOpen = new Set(open ? [...turn.trace.keys(), 'answer'] : []);
    this.render();
  }

  // --- Rendering ------------------------------------------------------------------------------

  render() {
    const turns = this.turns();
    if (!turns.includes(this.turn)) this.turn = turns.at(-1) || null;
    const turn = this.turn;
    this.picker.hidden = turns.length < 2;
    this.picker.replaceChildren(...turns.map((t, i) => h('option', { value: String(i) }, `Message ${i + 1}: ${clip(t.query, 48)}`)));
    if (turn) this.picker.value = String(turns.indexOf(turn));
    if (!turn) {
      this.summary.replaceChildren();
      this.body.replaceChildren(h('div', { class: 'log-empty' }, icon('log'),
        h('p', null, 'Send a message in the Test run tab. Every node it runs shows up here with its inputs and outputs.')));
      return;
    }
    turn.logOpen = turn.logOpen || new Set();
    const total = turn.trace.reduce((sum, s) => sum + (s.ms || 0), 0);
    const status = turn.running ? 'running' : turn.error ? 'failed' : 'done';
    this.summary.replaceChildren(
      h('span', { class: `log-status ${status}` }, { running: 'Running', failed: 'Failed', done: 'Finished' }[status]),
      ` · ${turn.trace.length} step${turn.trace.length === 1 ? '' : 's'} · ${formatMs(total)}`);

    const scroll = this.body.scrollTop;
    const failedStep = turn.trace.some((s) => s.status === 'failed');
    this.body.replaceChildren(
      ...turn.trace.map((step, i) => this.renderStep(turn, step, i)),
      turn.error && !failedStep ? h('div', { class: 'log-error' }, icon('alert'), h('span', null, turn.error)) : null,
      turn.running || turn.error ? null : this.renderAnswer(turn));
    this.body.scrollTop = scroll;
    this.highlight();
  }

  renderStep(turn, step, i) {
    const open = turn.logOpen.has(i);
    const info = this.editor.types[step.type];
    const tools = step.tools || [];
    const head = h('button', {
      type: 'button', class: 'log-step-head', 'aria-expanded': String(open), title: 'Show the details, and the node on the canvas',
      onClick: () => {
        if (open) turn.logOpen.delete(i); else turn.logOpen.add(i);
        if (this.editor.dsl.nodes[step.node]) this.editor.select({ type: 'node', id: step.node }, { keepTab: true });
        this.render();
      },
    },
    icon('chevronRight', 'caret'),
    h('span', { class: 'step-dot' }),
    h('span', { class: `node-icon small t-${step.type}` }, icon(step.type)),
    h('span', { class: 'log-node' }, step.node),
    h('span', { class: 'muted log-type' }, info?.title || step.type),
    tools.length ? h('span', { class: 'pill' }, `${tools.length} search${tools.length === 1 ? '' : 'es'}`) : null,
    step.status === 'done' && step.next !== undefined ? h('span', { class: 'muted log-next' }, `→ ${step.next || 'end'}`) : null,
    h('span', { class: 'step-ms' }, step.status === 'running' ? 'running…' : formatMs(step.ms)));
    const card = h('div', { class: `log-step ${step.status}${open ? ' open' : ''}`, dataset: { node: step.node } }, head);
    if (open) card.append(this.renderStepBody(step));
    return card;
  }

  renderStepBody(step) {
    const parts = [];
    if (step.error) parts.push(h('div', { class: 'log-error' }, icon('alert'), h('span', null, step.error)));
    const inputs = Object.entries(step.inputs || {});
    if (inputs.length) parts.push(logSection('Input', inputs.map(([k, v]) => this.entry(LABELS[k] || k, v, k))));
    if (step.tools?.length) parts.push(logSection('Searches the model made', step.tools.map((t, j) => this.toolCall(t, j))));
    const outputs = Object.entries(step.outputs || {});
    if (outputs.length) {
      parts.push(logSection('Output', outputs.map(([k, v]) => this.entry(`{${step.node}@${k}}`, v, k))));
    }
    if (!parts.length) parts.push(h('div', { class: 'muted' }, step.status === 'running' ? 'Running…' : 'Nothing was recorded.'));
    return h('div', { class: 'log-step-body' }, parts);
  }

  renderAnswer(turn) {
    const open = turn.logOpen.has('answer');
    const head = h('button', {
      type: 'button', class: 'log-step-head', 'aria-expanded': String(open),
      onClick: () => {
        if (open) turn.logOpen.delete('answer'); else turn.logOpen.add('answer');
        this.render();
      },
    }, icon('chevronRight', 'caret'), icon('message', 'log-answer-icon'), h('span', { class: 'log-node' }, 'Answer'),
    h('span', { class: 'muted log-type' }, `${turn.answer.length} characters · ${turn.references.length} reference${turn.references.length === 1 ? '' : 's'}`));
    const card = h('div', { class: `log-step answer${open ? ' open' : ''}` }, head);
    if (open) {
      card.append(h('div', { class: 'log-step-body' },
        this.entry('Answer', turn.answer || '(no message node ran)'),
        turn.references.length ? this.entry('References', turn.references, 'chunks') : null));
    }
    return card;
  }

  toolCall(call, index) {
    const args = call.arguments || {};
    const kind = !call.query ? 'Metadata only' : (args.filters?.length || args.sort_by) ? 'Query + metadata' : 'Query';
    return h('div', { class: `log-tool${call.error ? ' failed' : ''}` },
      h('div', { class: 'log-tool-head' },
        h('span', { class: 'log-tool-n' }, `#${index + 1}`),
        h('code', null, call.tool),
        h('span', { class: 'pill' }, kind),
        h('span', { class: 'muted' }, call.error ? 'error' : `${call.results} result${call.results === 1 ? '' : 's'}`),
        h('span', { class: 'step-ms' }, formatMs(call.elapsed_ms))),
      this.entry('Arguments', args),
      call.error ? h('div', { class: 'log-error' }, icon('alert'), h('span', null, `${call.error} (sent to the model, which can try again)`)) : null,
      call.result !== undefined ? this.entry('Result given to the model', call.result) : null);
  }

  // One labelled value. `key` picks a readable view for known values (messages, results, cases...).
  entry(label, value, key) {
    return h('div', { class: 'log-entry' }, h('div', { class: 'log-label' }, label), this.view(value, key));
  }

  view(value, key) {
    if (key === 'kb_ids' && Array.isArray(value)) return inline(value.map((id) => `${this.editor.kbName(id)} (#${id})`).join(', '));
    if (key === 'messages' && Array.isArray(value)) return messagesView(value);
    if (key === 'history' && Array.isArray(value)) return value.length ? messagesView(value) : muted('none');
    if (key === 'chunks' && Array.isArray(value)) return value.length ? h('div', { class: 'log-refs' }, value.map(refView)) : muted('none');
    if (key === 'filters' && Array.isArray(value)) {
      return value.length ? inline(value.map((f) => `${f.field} ${f.op} ${JSON.stringify(f.value)}`).join('  and  ')) : muted('none');
    }
    if (key === 'cases' && Array.isArray(value)) return casesView(value);
    if (key === 'skipped_filters' && Array.isArray(value)) return inline(value.join('  and  '));
    if (key === 'tools' && Array.isArray(value)) return h('div', { class: 'log-tools' }, value.map(toolView));
    if (key === 'model' && value && typeof value === 'object') return inline(Object.values(value).join(' · '));
    return valueView(value);
  }

  // --- Export -------------------------------------------------------------------------------

  json(turn) {
    return {
      query: turn.query,
      inputs: turn.inputs || {},
      status: turn.running ? 'running' : turn.error ? 'failed' : 'done',
      error: turn.error || undefined,
      steps: turn.trace.map((s) => ({
        node: s.node, type: s.type, status: s.status, elapsed_ms: s.ms, next: s.next,
        inputs: s.inputs, tool_calls: s.tools, outputs: s.outputs, error: s.error,
      })),
      answer: turn.answer,
      references: turn.references,
    };
  }

  async copy() {
    if (!this.turn) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(this.json(this.turn), null, 2));
      toast('Copied the run log as JSON');
    } catch {
      toast('Could not copy (the clipboard needs HTTPS or localhost). Use Download.', 'error');
    }
  }

  download() {
    if (!this.turn) return;
    const n = this.turns().indexOf(this.turn) + 1;
    downloadJson(`run-log-${n}.json`, this.json(this.turn));
  }

  // --- Size ------------------------------------------------------------------------------------

  startResize(e) {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = this.el.offsetHeight;
    const max = () => Math.max(160, (this.el.parentElement?.clientHeight || 600) - 90);
    const move = (ev) => {
      this.height = Math.round(Math.min(max(), Math.max(120, startHeight + startY - ev.clientY)));
      this.applyHeight();
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      try { localStorage.setItem(HEIGHT_KEY, String(this.height)); } catch { /* storage blocked */ }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }

  applyHeight() {
    this.el.style.height = `${this.height}px`;
  }
}

// --- Views --------------------------------------------------------------------------------------

function loadHeight() {
  try {
    const n = Number(localStorage.getItem(HEIGHT_KEY));
    if (n >= 120 && n <= 2000) return n;
  } catch { /* storage blocked */ }
  return 320;
}

function logSection(title, children) {
  return h('div', { class: 'log-section' }, h('div', { class: 'log-section-title' }, title), children);
}

const muted = (text) => h('span', { class: 'muted' }, text);
const inline = (text) => h('code', { class: 'log-inline' }, text);

function valueView(value) {
  if (value === null || value === undefined) return muted('null');
  if (typeof value === 'string') {
    if (!value) return muted('(empty)');
    if (value.length <= 90 && !value.includes('\n')) return inline(value);
    return textBlock(value);
  }
  if (typeof value !== 'object') return inline(String(value));
  const compact = JSON.stringify(value);
  if (compact.length <= 90) return inline(compact);
  return textBlock(JSON.stringify(value, null, 2), 'json');
}

function textBlock(text, kind = '') {
  return h('div', { class: 'log-block' },
    h('pre', { class: `log-text ${kind}` }, text),
    iconButton('copy', 'Copy', () => copyText(text), { class: 'log-copy' }));
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied');
  } catch {
    toast('Could not copy (the clipboard needs HTTPS or localhost)', 'error');
  }
}

function messagesView(messages) {
  return h('div', { class: 'log-messages' }, messages.map((m) => h('div', { class: `log-message ${m.role}` },
    h('div', { class: 'log-role' }, m.role),
    typeof m.content === 'string' ? h('div', { class: 'log-message-text' }, m.content) : valueView(m.content))));
}

function refView(r) {
  const metadata = Object.entries(r.metadata || {}).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join(' · ');
  return h('div', { class: `ref${r.cited ? ' cited' : ''}` },
    h('div', { class: 'ref-head' },
      h('span', { class: 'ref-n' }, `[${r.ref}]`),
      h('span', { class: 'ref-title' }, r.title || r.external_id),
      r.cited ? h('span', { class: 'pill' }, 'cited') : null,
      h('span', { class: 'muted' }, r.score === null || r.score === undefined ? 'by metadata' : `score ${r.score}`)),
    metadata ? h('div', { class: 'ref-meta' }, metadata) : null,
    h('div', { class: 'ref-body' }, r.content));
}

function casesView(cases) {
  if (!cases.length) return muted('none');
  return h('div', { class: 'log-cases' }, cases.map((c) => h('div', { class: `log-case${c.matched ? ' matched' : ''}` },
    h('div', { class: 'log-case-head' }, `Case ${c.case + 1}`, c.conditions.length > 1 ? ` (${c.logic})` : '',
      h('span', { class: c.matched ? 'log-yes' : 'muted' }, c.matched ? ' → matched' : ' → no match')),
    c.conditions.map((x) => h('div', { class: 'log-condition' },
      h('code', null, x.var), ' = ', inline(JSON.stringify(x.value) ?? 'null'), ` ${x.op} `,
      ['empty', 'not_empty'].includes(x.op) ? '' : inline(JSON.stringify(x.compared_to) ?? 'null'),
      h('span', { class: x.result ? 'log-yes' : 'log-no' }, x.result ? '  ✓' : '  ✗'))))));
}

function toolView(tool) {
  return h('details', { class: 'log-tool-spec' },
    h('summary', null, icon('chevronRight', 'caret'), h('code', null, tool.name)),
    h('div', { class: 'log-entry' }, h('div', { class: 'log-label' }, 'Description'), valueView(tool.description)),
    tool.fixed_filters ? h('div', { class: 'log-entry' }, h('div', { class: 'log-label' }, 'Fixed filters'),
      inline(tool.fixed_filters.map((f) => `${f.field} ${f.op} ${JSON.stringify(f.value)}`).join('  and  '))) : null,
    tool.skipped_filters ? h('div', { class: 'log-entry' }, h('div', { class: 'log-label' }, 'Filters skipped (empty value)'),
      inline(tool.skipped_filters.join('  and  '))) : null,
    h('div', { class: 'log-entry' }, h('div', { class: 'log-label' }, 'Arguments (JSON schema)'), textBlock(JSON.stringify(tool.parameters, null, 2), 'json')));
}

function formatMs(ms) {
  if (ms === undefined || ms === null) return '';
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function clip(text, n) {
  const t = String(text || '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n)}…` : t;
}

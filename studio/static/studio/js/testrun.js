// The test panel: chat with the workflow as it is in the editor (unsaved changes included).
// Nothing is stored on the server; the conversation lives here and is sent back as history.
// Test runs are debug runs: each node's inputs and outputs go to the run log (runlog.js).

import { ApiError, stream } from './api.js';
import { beginId } from './graph.js';
import { autoGrow, button, formatLoose, h, icon, iconButton, parseLoose } from './ui.js';

export class TestPanel {
  constructor(editor) {
    this.editor = editor;
    this.turns = []; // {query, inputs, answer, trace: [step], references: [], error, running}
    // step: {node, type, status: running|done|failed, ms, inputs, outputs, next, tools: [], error}
    this.inputs = {}; // overrides for the begin node's inputs, as typed
    this.controller = null;
    this.el = h('div', { class: 'test-panel' });
    this.messages = h('div', { class: 'messages' });
    this.input = autoGrow(h('textarea', {
      rows: 1, placeholder: 'Send a message to test the workflow…', 'aria-label': 'Test message',
      onKeydown: (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); }
      },
    }), 160);
    this.sendButton = iconButton('send', 'Send (Enter)', () => this.send(), { class: 'send' });
    this.stopButton = iconButton('stop', 'Stop', () => this.controller?.abort(), { class: 'send stop' });
    this.inputsBox = h('div', { class: 'test-inputs' });
    this.el.append(
      h('div', { class: 'test-head' },
        h('div', null, h('div', { class: 'panel-type' }, 'Test run'),
          h('div', { class: 'panel-desc' }, 'Runs the workflow as it is in the editor, unsaved changes included. Nothing is saved.')),
        h('div', { class: 'test-actions' },
          button('Run log', { icon: 'log', kind: 'small', title: 'Every node of a run with its inputs and outputs', onClick: () => this.editor.runLog.toggle() }),
          button('New chat', { icon: 'refresh', kind: 'small', onClick: () => this.reset() }))),
      this.inputsBox,
      this.messages,
      h('div', { class: 'composer' }, this.input, this.sendButton, this.stopButton));
    this.render();
  }

  reset() {
    this.runId = (this.runId || 0) + 1; // the running request, if any, is now stale
    this.controller?.abort();
    this.turns = [];
    this.editor.setRunState({});
    this.editor.runLog.reset();
    this.render();
    this.input.focus();
  }

  // The begin node's inputs, with the values typed here.
  renderInputs() {
    const begin = this.editor.dsl.nodes[beginId(this.editor.dsl)];
    const defaults = begin?.params?.inputs || {};
    const names = Object.keys(defaults);
    if (!names.length) {
      this.inputsBox.replaceChildren();
      return;
    }
    this.inputsBox.replaceChildren(h('details', { class: 'group', open: true },
      h('summary', null, icon('chevronRight', 'caret'), 'Inputs'),
      h('div', { class: 'group-body' }, names.map((name) => h('div', { class: 'kv-row' },
        h('code', { class: 'input-name' }, name),
        h('input', {
          type: 'text', value: this.inputs[name] ?? formatLoose(defaults[name]), spellcheck: 'false', 'aria-label': name,
          onInput: (e) => { this.inputs[name] = e.target.value; },
        })))),
      h('div', { class: 'help' }, 'Sent as "inputs" with each run, like your app would.')));
  }

  inputValues() {
    const begin = this.editor.dsl.nodes[beginId(this.editor.dsl)];
    const out = {};
    for (const name of Object.keys(begin?.params?.inputs || {})) {
      if (this.inputs[name] !== undefined) out[name] = parseLoose(this.inputs[name]);
    }
    return out;
  }

  render() {
    this.renderInputs();
    const running = this.turns.some((t) => t.running);
    this.sendButton.hidden = running;
    this.stopButton.hidden = !running;
    if (!this.turns.length) {
      this.messages.replaceChildren(h('div', { class: 'test-empty' }, icon('play'), h('p', null, 'Ask something your users would ask. Nodes light up on the canvas as they run.')));
      return;
    }
    const m = this.messages;
    const atBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 60;
    const scroll = m.scrollTop;
    m.replaceChildren(...this.turns.map((t) => this.renderTurn(t)));
    m.scrollTop = atBottom ? m.scrollHeight : scroll;
  }

  renderTurn(turn) {
    const answer = h('div', { class: 'bubble assistant' });
    if (turn.answer) answer.append(withCitations(turn.answer));
    else if (turn.running) answer.append(h('span', { class: 'typing' }, h('i'), h('i'), h('i')));
    else if (!turn.error) answer.append(h('span', { class: 'muted' }, 'No answer (the run ended without a message node).'));
    if (turn.error) answer.append(h('div', { class: 'run-error' }, icon('alert'), h('span', null, turn.error)));
    turn.answerEl = answer;

    const total = turn.trace.reduce((sum, s) => sum + (s.ms || 0), 0);
    const failed = turn.trace.some((s) => s.status === 'failed');
    const log = turn.trace.length ? h('button', {
      type: 'button', class: `log-link${failed ? ' failed' : ''}`, title: 'Every node of this run with its inputs and outputs',
      onClick: () => this.editor.runLog.toggle(turn),
    }, icon('log'), `Run log · ${turn.trace.length} step${turn.trace.length === 1 ? '' : 's'} · ${formatMs(total)}`,
    failed ? ' · failed' : '', icon('chevronRight')) : null;

    const refs = turn.references.length ? h('details', { class: 'trace' },
      h('summary', null, icon('chevronRight', 'caret'), `References · ${turn.references.length}`),
      turn.references.map((r) => h('div', { class: `ref${r.cited ? ' cited' : ''}` },
        h('div', { class: 'ref-head' }, h('span', { class: 'ref-n' }, `[${r.ref}]`), h('span', { class: 'ref-title' }, r.title || r.external_id),
          r.cited ? h('span', { class: 'pill' }, 'cited') : null,
          h('span', { class: 'muted' }, r.score === null || r.score === undefined ? 'by metadata' : `score ${r.score}`)),
        h('div', { class: 'ref-body' }, r.content.length > 280 ? `${r.content.slice(0, 280)}…` : r.content)))) : null;
    if (refs) {
      refs.open = !!turn.refsOpen;
      refs.addEventListener('toggle', () => { turn.refsOpen = refs.open; });
    }

    return h('div', { class: 'turn' },
      h('div', { class: 'bubble user' }, turn.query),
      answer, log, refs);
  }

  async send() {
    const query = this.input.value.trim();
    if (!query || this.turns.some((t) => t.running)) return;
    this.input.value = '';
    this.input.dispatchEvent(new Event('input'));
    const history = [];
    for (const t of this.turns) {
      if (t.error) continue;
      history.push({ role: 'user', content: t.query }, { role: 'assistant', content: t.answer || '' });
    }
    const inputs = this.inputValues();
    const turn = { query, inputs, answer: '', trace: [], references: [], error: null, running: true };
    this.turns.push(turn);
    const runState = {};
    this.editor.setRunState(runState);
    this.render();
    this.editor.runLog.follow(turn);

    const run = this.runId = (this.runId || 0) + 1;
    const controller = this.controller = new AbortController();
    try {
      await stream('workflows/test-run', { dsl: this.editor.payloadDsl(), query, history: history.slice(-50), inputs }, (event) => {
        this.onEvent(turn, runState, event);
      }, controller.signal);
    } catch (err) {
      if (run !== this.runId || this.editor.destroyed) return;
      if (err.name === 'AbortError') turn.error = 'Stopped.';
      else if (err instanceof ApiError && err.errors.length) {
        turn.error = 'Fix the problems in the workflow first.';
        this.editor.setErrors(err.errors);
      } else turn.error = err.message;
      for (const [id, s] of Object.entries(runState)) if (s.status === 'running') runState[id] = { status: 'failed', error: turn.error };
      for (const step of turn.trace) if (step.status === 'running') Object.assign(step, { status: 'failed', error: turn.error });
    } finally {
      turn.running = false;
      if (run === this.runId && !this.editor.destroyed) {
        this.controller = null;
        this.editor.setRunState(runState);
        this.render();
        this.editor.runLog.update(turn);
      }
    }
  }

  onEvent(turn, runState, event) {
    if (!this.turns.includes(turn)) return; // from a run that "New chat" dropped
    const running = (node) => [...turn.trace].reverse().find((s) => s.node === node && s.status === 'running');
    switch (event.event) {
      case 'node_started':
        turn.trace.push({ node: event.node, type: event.type, status: 'running', tools: [] });
        runState[event.node] = { status: 'running' };
        break;
      case 'node_finished': {
        const step = running(event.node);
        if (step) Object.assign(step, { status: 'done', ms: event.elapsed_ms, inputs: event.inputs, outputs: event.outputs, next: event.next });
        runState[event.node] = { status: 'done', ms: event.elapsed_ms };
        break;
      }
      case 'node_failed': {
        const step = running(event.node);
        if (step) Object.assign(step, { status: 'failed', ms: event.elapsed_ms, inputs: event.inputs, error: event.error });
        runState[event.node] = { status: 'failed', error: event.error };
        break;
      }
      case 'tool_call': {
        const step = [...turn.trace].reverse().find((s) => s.node === event.node);
        if (step) step.tools.push(event);
        break;
      }
      case 'message':
        turn.answer += event.delta;
        if (turn.answerEl) {
          turn.answerEl.replaceChildren(withCitations(turn.answer));
          this.messages.scrollTop = this.messages.scrollHeight;
        }
        return; // the log shows the answer when the run is done
      case 'done':
        turn.answer = event.answer;
        turn.references = event.references || [];
        break;
      case 'error': {
        turn.error = event.message;
        const step = [...turn.trace].reverse().find((s) => s.status === 'running');
        if (step) {
          Object.assign(step, { status: 'failed', ms: 0, error: event.message });
          runState[step.node] = { status: 'failed', error: event.message };
        }
        break;
      }
      default:
        return;
    }
    this.editor.setRunState(runState);
    this.render();
    this.editor.runLog.update(turn);
  }
}

function formatMs(ms) {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

// Answer text with [n] citations as small badges.
function withCitations(text) {
  const out = h('span', { class: 'answer-text' });
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
    if (/^\[\d+\]$/.test(part)) out.append(h('sup', { class: 'cite' }, part.slice(1, -1)));
    else if (part) out.append(part);
  }
  return out;
}

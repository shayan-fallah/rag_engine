// The home page: all workflows, and creating one (blank, from a template, or imported).

import { api, config } from './api.js';
import { button, clone, confirmDialog, downloadJson, h, icon, iconButton, modal, slug, timeAgo, toast } from './ui.js';

const BLANK = {
  key: 'blank',
  name: 'Blank',
  description: 'Begin → LLM → Message: a plain chat with the model. Add nodes from there.',
  dsl: {
    nodes: {
      begin: { type: 'begin', next: 'answer' },
      answer: { type: 'llm', params: { system_prompt: 'You are a helpful assistant.', history: 6 }, next: 'reply' },
      reply: { type: 'message', params: { text: '{answer@content}' } },
    },
  },
};

// Does the workflow search knowledge bases?
function usesKbs(dsl) {
  return Object.values(dsl.nodes || {}).some((n) => n.type === 'retrieval' || n.type === 'agent');
}

// Point every search of a workflow at the given knowledge bases.
function setKbs(dsl, ids) {
  for (const node of Object.values(dsl.nodes || {})) {
    if (node.type === 'retrieval') node.params = { ...node.params, kb_ids: [...ids] };
    if (node.type === 'agent') for (const tool of node.params?.tools || []) tool.kb_ids = [...ids];
  }
  return dsl;
}

export function header() {
  return h('header', { class: 'topbar topbar-home' },
    h('a', { class: 'brand', href: '#/' }, icon('logo'), h('span', null, 'RAG Engine'), h('span', { class: 'brand-sub' }, 'Studio')),
    h('span', { class: 'spacer' }),
    h('a', { class: 'top-link', href: config.docsUrl, target: '_blank', rel: 'noopener' }, 'API docs', icon('external')),
    h('a', { class: 'top-link', href: config.adminUrl }, 'Admin'),
    h('span', { class: 'user' }, config.user),
    iconButton('logout', 'Log out', () => document.getElementById('logout-form').submit()));
}

export class ListPage {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.workflows = [];
    this.templates = [];
    root.replaceChildren(h('div', { class: 'home-page' }, header(), h('div', { class: 'boot' }, 'Loading…')));
    this.load();
  }

  destroy() {
    this.destroyed = true;
  }

  async load() {
    try {
      const [workflows, templates, kbs] = await Promise.all([api.get('workflows'), api.get('workflow-templates'), api.get('kbs')]);
      this.workflows = workflows;
      this.templates = templates;
      this.app.kbs = kbs;
    } catch (err) {
      if (this.destroyed) return;
      this.root.replaceChildren(h('div', { class: 'home-page' }, header(), h('div', { class: 'boot error' }, h('h1', null, 'Could not load workflows'), h('p', null, err.message))));
      return;
    }
    if (!this.destroyed) this.render();
  }

  render() {
    const sorted = [...this.workflows].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    const list = sorted.length
      ? h('div', { class: 'wf-grid' }, sorted.map((wf) => this.card(wf)))
      : h('div', { class: 'empty-state' },
        icon('workflow'),
        h('h2', null, 'No workflows yet'),
        h('p', null, 'A workflow decides how your app answers: search the knowledge base, route questions, call the model. Start from a template or a blank one.'),
        button('New workflow', { icon: 'plus', kind: 'primary', onClick: () => this.newWorkflow() }));
    this.root.replaceChildren(h('div', { class: 'home-page' }, header(),
      h('main', { class: 'home' },
        h('div', { class: 'home-head' },
          h('div', null, h('h1', null, 'Workflows'), h('p', { class: 'muted' }, 'Build them visually here, call them from your app with POST /api/workflows/{id}/run.')),
          h('div', { class: 'home-actions' },
            button('Import', { icon: 'upload', onClick: () => this.importWorkflow() }),
            button('New workflow', { icon: 'plus', kind: 'primary', onClick: () => this.newWorkflow() }))),
        this.kbNote(),
        list)));
  }

  kbNote() {
    if (this.app.kbs.length) return null;
    return h('div', { class: 'notice' }, icon('alert'),
      h('span', null, 'There are no knowledge bases yet, so retrieval and agent nodes have nothing to search. Create one with POST /api/kbs (see the API docs), then add documents or a data source.'));
  }

  card(wf) {
    const nodes = Object.values(wf.dsl.nodes || {});
    const types = [...new Set(nodes.map((n) => n.type))].filter((t) => t !== 'begin');
    return h('div', { class: 'wf-card', role: 'link', tabindex: '0', onClick: () => { location.hash = `#/w/${wf.id}`; }, onKeydown: (e) => { if (e.key === 'Enter') location.hash = `#/w/${wf.id}`; } },
      h('div', { class: 'wf-card-head' },
        h('div', { class: 'wf-name' }, wf.name),
        h('div', { class: 'wf-card-actions', onClick: (e) => e.stopPropagation() },
          iconButton('copy', 'Duplicate', () => this.duplicate(wf)),
          iconButton('download', 'Export as JSON', () => downloadJson(`${slug(wf.name)}.json`, { name: wf.name, description: wf.description, dsl: wf.dsl })),
          iconButton('trash', 'Delete', () => this.remove(wf)))),
      h('div', { class: 'wf-desc' }, wf.description || h('span', { class: 'muted' }, 'No description')),
      h('div', { class: 'wf-meta' },
        h('span', { class: 'type-dots' }, types.map((t) => h('span', { class: `type-dot t-${t}`, title: this.app.types[t]?.title || t }, icon(t)))),
        h('span', null, `${nodes.length} nodes · #${wf.id} · ${timeAgo(wf.updated_at)}`)));
  }

  uniqueName(base) {
    const names = new Set(this.workflows.map((w) => w.name));
    if (!names.has(base)) return base;
    for (let i = 2; ; i++) if (!names.has(`${base} ${i}`)) return `${base} ${i}`;
  }

  kbPicker(selected, onChange) {
    const kbs = this.app.kbs;
    if (!kbs.length) return h('div', { class: 'empty-note' }, 'No knowledge bases yet, and this workflow searches one. Create one first with POST /api/kbs (see the API docs), or start from Blank.');
    return h('div', { class: 'checklist' }, kbs.map((kb) => h('label', { class: 'check' },
      h('input', {
        type: 'checkbox', checked: selected.includes(kb.id),
        onChange: (e) => onChange(e.target.checked ? [...selected, kb.id] : selected.filter((x) => x !== kb.id)),
      }),
      h('span', { class: 'check-label' }, kb.name), h('span', { class: 'muted' }, `${kb.document_count} docs`))));
  }

  newWorkflow() {
    const choices = [BLANK, ...this.templates];
    let choice = BLANK;
    let kbIds = this.app.kbs.length ? [this.app.kbs[0].id] : [];
    let nameTouched = false;
    const name = h('input', { type: 'text', maxlength: 200, value: this.uniqueName('New workflow'), onInput: () => { nameTouched = true; } });
    const description = h('textarea', { rows: 2, placeholder: 'What it is for (optional)' });
    const kbBox = h('div', { class: 'field' });
    const error = h('div', { class: 'form-error' });
    const drawKbs = () => {
      kbBox.hidden = !usesKbs(choice.dsl);
      kbBox.replaceChildren(h('div', { class: 'field-label' }, h('label', null, 'Knowledge bases to search')),
        this.kbPicker(kbIds, (ids) => { kbIds = ids; drawKbs(); }));
    };
    const options = h('div', { class: 'choices' }, choices.map((c) => h('label', { class: 'choice' },
      h('input', {
        type: 'radio', name: 'start', checked: c === choice,
        onChange: () => {
          choice = c;
          if (!nameTouched) name.value = this.uniqueName(c === BLANK ? 'New workflow' : c.name);
          if (!description.value || choices.some((x) => x.description === description.value)) description.value = c === BLANK ? '' : c.description || '';
          drawKbs();
        },
      }),
      h('span', { class: 'choice-text' }, h('span', { class: 'choice-name' }, c.name), h('span', { class: 'choice-desc' }, c.description)))));
    drawKbs();
    modal({
      title: 'New workflow',
      body: h('div', { class: 'form' },
        h('div', { class: 'field' }, h('div', { class: 'field-label' }, h('label', null, 'Start from')), options),
        kbBox,
        h('div', { class: 'field' }, h('div', { class: 'field-label' }, h('label', null, 'Name')), name),
        h('div', { class: 'field' }, h('div', { class: 'field-label' }, h('label', null, 'Description')), description),
        error),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Create', kind: 'primary',
          onClick: async () => {
            const dsl = clone(choice.dsl);
            if (usesKbs(dsl)) setKbs(dsl, kbIds);
            return this.create({ name: name.value.trim(), description: description.value.trim(), dsl }, error);
          },
        },
      ],
    });
  }

  async create(body, errorEl) {
    if (!body.name) {
      if (errorEl) errorEl.textContent = 'Give it a name.';
      return false;
    }
    try {
      const wf = await api.post('workflows', body);
      location.hash = `#/w/${wf.id}`;
      return true;
    } catch (err) {
      const message = err.errors?.length ? err.errors.map((e) => (e.node ? `${e.node}: ${e.message}` : e.message)).join('\n') : err.message;
      if (errorEl) errorEl.textContent = message;
      else toast(message, 'error');
      return false;
    }
  }

  importWorkflow() {
    const text = h('textarea', { class: 'prompt code', rows: 10, spellcheck: 'false', placeholder: 'Paste an exported workflow ({"name", "description", "dsl"}) or just its "dsl" ({"nodes": ...})' });
    const file = h('input', {
      type: 'file', accept: '.json,application/json',
      onChange: async () => { if (file.files[0]) text.value = await file.files[0].text(); },
    });
    const name = h('input', { type: 'text', maxlength: 200, placeholder: 'Taken from the file if empty' });
    let kbIds = [];
    let replaceKbs = false;
    const kbBox = h('div');
    const drawKbs = () => {
      kbBox.replaceChildren(
        h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: replaceKbs, onChange: (e) => { replaceKbs = e.target.checked; drawKbs(); } }),
          h('span', { class: 'check-label' }, 'Point every search at knowledge bases of this server')),
        replaceKbs ? this.kbPicker(kbIds, (ids) => { kbIds = ids; drawKbs(); }) : null,
        h('div', { class: 'help' }, 'Useful when the workflow comes from another server, where knowledge base ids differ.'));
    };
    drawKbs();
    const error = h('div', { class: 'form-error' });
    modal({
      title: 'Import a workflow',
      wide: true,
      body: h('div', { class: 'form' }, file, text,
        h('div', { class: 'field' }, h('div', { class: 'field-label' }, h('label', null, 'Name')), name),
        kbBox, error),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Import', kind: 'primary',
          onClick: async () => {
            let data;
            try {
              data = JSON.parse(text.value);
            } catch (e) {
              error.textContent = `Not valid JSON: ${e.message}`;
              return false;
            }
            const dsl = data?.dsl || data;
            if (!dsl || typeof dsl.nodes !== 'object') {
              error.textContent = 'No workflow found: expected {"dsl": {"nodes": ...}} or {"nodes": ...}.';
              return false;
            }
            if (replaceKbs) setKbs(dsl, kbIds);
            return this.create({
              name: name.value.trim() || this.uniqueName(data.name || 'Imported workflow'),
              description: data.description || '',
              dsl,
            }, error);
          },
        },
      ],
    });
  }

  async duplicate(wf) {
    try {
      const copy = await api.post('workflows', { name: this.uniqueName(`${wf.name} (copy)`), description: wf.description, dsl: wf.dsl });
      this.workflows.push(copy);
      if (!this.destroyed) this.render();
      toast(`Created ${copy.name}`);
    } catch (err) {
      toast(err.message, 'error', 6000);
    }
  }

  async remove(wf) {
    const ok = await confirmDialog('Delete this workflow?', `"${wf.name}" and its conversations will be deleted. Apps that use it will get errors.`, { confirm: 'Delete', danger: true });
    if (!ok) return;
    try {
      await api.del(`workflows/${wf.id}`);
      this.workflows = this.workflows.filter((w) => w.id !== wf.id);
      if (!this.destroyed) this.render();
      toast('Workflow deleted');
    } catch (err) {
      toast(err.message, 'error');
    }
  }
}

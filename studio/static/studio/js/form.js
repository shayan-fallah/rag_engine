// Settings forms, built from the JSON schema of each node type's params (GET /api/node-types).
// Every param of the engine shows up here; json_schema_extra hints pick the widget.

import { NODE_ID, ROUTERS, nodeOutputs, outPorts, variablesFor } from './graph.js';
import {
  autoGrow, button, clone, closePopover, deepEqual, formatLoose, h, icon, iconButton,
  insertAtCursor, modal, parseLoose, popover, toast,
} from './ui.js';

const GROUP_TITLES = { model: 'Model', metadata: 'Metadata search', advanced: 'Advanced' };
// Field ids follow the setting's path, so the editor can put the focus back after redrawing the panel.
const fieldId = (path) => `f-${path.replace(/[^A-Za-z0-9_-]/g, '-')}`;

// --- Schema helpers -------------------------------------------------------------------

function deref(schema, root) {
  while (schema && schema.$ref) schema = root.$defs?.[schema.$ref.split('/').pop()];
  return schema || {};
}

function describe(name, prop, root, required) {
  let inner = prop;
  let nullable = false;
  if (prop.anyOf) {
    const rest = prop.anyOf.filter((s) => s.type !== 'null');
    nullable = rest.length < prop.anyOf.length;
    inner = rest.length === 1 ? rest[0] : {};
  }
  inner = deref(inner, root);
  return { name, prop, inner, nullable, required, title: prop.title || name, widget: widgetOf(prop, inner, root) };
}

function widgetOf(prop, inner, root) {
  if (prop.widget) return prop.widget;
  if (inner.enum) return 'enum';
  if (inner.type === 'boolean') return 'bool';
  if (inner.type === 'integer' || inner.type === 'number') return 'number';
  if (inner.type === 'array') return deref(inner.items, root).type === 'object' ? 'objects' : 'strings';
  if (inner.type === 'object') return 'kv';
  if (inner.type === 'string') return 'string';
  return 'value';
}

const isEmpty = (v) => v == null || v === '' || (Array.isArray(v) && !v.length) || (typeof v === 'object' && !Array.isArray(v) && !Object.keys(v).length);

function currentValue(obj, f) {
  if (obj[f.name] !== undefined) return obj[f.name];
  if (f.prop.default !== undefined) return clone(f.prop.default);
  if (f.inner.type === 'array') return [];
  if (f.inner.type === 'object') return f.nullable ? null : {};
  return f.nullable ? null : f.inner.type === 'string' ? '' : undefined;
}

// Values equal to the default are left out of the workflow, so it stays short and gets new defaults.
function storeValue(obj, f, value) {
  const isDefault = f.prop.default !== undefined ? deepEqual(value, f.prop.default) : isEmpty(value);
  if (!f.required && isDefault) delete obj[f.name];
  else obj[f.name] = value;
}

function singular(title) {
  const t = title.toLowerCase();
  if (t.endsWith('ies')) return `${t.slice(0, -3)}y`;
  if (t.endsWith('s')) return t.slice(0, -1);
  return t;
}

// A new list item: required fields get starting values.
function newItem(schema, root, f, list, ctx) {
  const item = {};
  const used = new Set(list.map((x) => x?.name));
  for (const name of schema.required || []) {
    const sub = describe(name, schema.properties[name], root, true);
    if (name === 'name') {
      const base = f.name === 'tools' ? 'search' : f.name === 'categories' ? 'category' : 'item';
      let i = list.length + 1;
      while (used.has(`${base}_${i}`)) i++;
      item.name = `${base}_${i}`;
    } else if (sub.widget === 'kbs') item[name] = ctx.editor.kbs.length ? [ctx.editor.kbs[0].id] : [];
    else if (sub.widget === 'objects') item[name] = [newItem(deref(sub.inner.items, root), root, sub, [], ctx)];
    else if (sub.widget === 'template') item[name] = '{sys.query}';
    else if (sub.inner.enum) item[name] = sub.inner.enum[0];
    else if (sub.inner.type === 'array') item[name] = [];
    else if (sub.inner.type === 'integer' || sub.inner.type === 'number') item[name] = sub.prop.default ?? sub.inner.minimum ?? 0;
    else item[name] = '';
  }
  return item;
}

// --- Rendering --------------------------------------------------------------------------

// The fields of `obj` (a node's params or a list item), grouped like the schema says.
function renderFields(obj, schema, root, ctx, path) {
  const required = new Set(schema.required || []);
  const all = Object.entries(schema.properties || {}).map(([name, prop]) => describe(name, prop, root, required.has(name)));
  // A change that shows or hides other fields (e.g. "Find by") redraws the panel.
  const visibility = (o) => all.map((f) => hidden(f, o)).join();
  ctx = { ...ctx, visibility, kbIds: Array.isArray(obj.kb_ids) ? obj.kb_ids : ctx.kbIds };
  const fields = all.filter((f) => !hidden(f, obj));
  const out = fields.filter((f) => !f.prop.group).map((f) => renderField(obj, f, root, ctx, path));
  const groups = {};
  for (const f of fields) if (f.prop.group) (groups[f.prop.group] = groups[f.prop.group] || []).push(f);
  for (const [group, list] of Object.entries(groups)) {
    const key = `${ctx.nodeId}:${path}:${group}`;
    const changed = list.filter((f) => obj[f.name] !== undefined).length;
    const details = h('details', { class: 'group', open: ctx.editor.openGroups.has(key) },
      h('summary', null, icon('chevronRight', 'caret'), GROUP_TITLES[group] || group,
        changed ? h('span', { class: 'changed-count', title: 'Settings changed from the defaults' }, `${changed} changed`) : null),
      h('div', { class: 'group-body' }, list.map((f) => renderField(obj, f, root, ctx, path))));
    details.addEventListener('toggle', () => {
      if (details.open) ctx.editor.openGroups.add(key); else ctx.editor.openGroups.delete(key);
    });
    out.push(details);
  }
  return out;
}

// hide_when: {param: [values]} or {param: "empty"}; hidden when any of them holds.
function hidden(f, obj) {
  const when = f.prop.hide_when;
  return !!when && Object.entries(when).some(([key, values]) => (values === 'empty' ? isEmpty(obj[key]) : values.includes(obj[key])));
}

function renderField(obj, f, root, ctx, path) {
  const id = fieldId(`${path}.${f.name}`);
  const key = `${ctx.nodeId}:${path}.${f.name}`;
  const value = currentValue(obj, f);
  const set = (v, { structural = false } = {}) => {
    const trial = { ...obj };
    storeValue(trial, f, v);
    const toggles = ctx.visibility && ctx.visibility(trial) !== ctx.visibility(obj);
    ctx.change(key, () => storeValue(obj, f, v), structural || toggles);
  };
  // "Reset" only where the default is something (a prompt, a number), not for empty defaults.
  const customized = obj[f.name] !== undefined && f.prop.default !== undefined && !isEmpty(f.prop.default)
    && !deepEqual(obj[f.name], f.prop.default);
  const widget = WIDGETS[f.widget] || WIDGETS.value;
  const control = widget({ value, set, f, ctx, id, root, obj, path });
  return h('div', { class: `field w-${f.widget}${customized ? ' customized' : ''}` },
    h('div', { class: 'field-label' },
      h('label', { for: id }, f.title),
      customized ? h('button', {
        type: 'button', class: 'link-btn', title: 'Back to the default value',
        onClick: () => ctx.change(key, () => { delete obj[f.name]; }, true),
      }, 'Reset') : null),
    control,
    f.prop.description ? h('div', { class: 'help' }, f.prop.description) : null);
}

function placeholderFor(f, ctx, obj) {
  const d = ctx.editor.modelDefaults;
  const provider = obj.provider || d.provider;
  switch (f.name) {
    case 'model': return `Default: ${provider === 'anthropic' ? d.anthropic_model : d.openai_model}`;
    case 'temperature': return 'Model default';
    case 'max_tokens': return provider === 'anthropic' ? `Default: ${d.max_tokens}` : 'Model default';
    case 'max_distance': return 'No limit';
    default: return f.nullable ? 'Default' : '';
  }
}

function enumDefaultLabel(f, ctx) {
  const d = ctx.editor.modelDefaults;
  if (f.name === 'provider') return `Default (${d.provider})`;
  if (f.name === 'effort') return `Default (${d.effort || 'model default'})`;
  return 'Default';
}

const OP_LABELS = {
  '==': 'equals', '!=': 'does not equal', '>': 'greater than', '>=': 'at least', '<': 'less than', '<=': 'at most',
  contains: 'contains', not_contains: 'does not contain', empty: 'is empty', not_empty: 'is not empty', in: 'is one of',
};

const WIDGETS = {
  string({ value, set, f, ctx, id, obj }) {
    return h('input', {
      id, type: 'text', value: value ?? '', placeholder: f.prop.placeholder || placeholderFor(f, ctx, obj),
      maxlength: f.inner.maxLength, spellcheck: 'false',
      onInput: (e) => set(f.nullable && !e.target.value ? null : e.target.value),
    });
  },

  // A metadata field name, with the fields of the knowledge bases as suggestions.
  field({ value, set, f, ctx, id, obj }) {
    const listId = `${id}-fields`;
    const list = h('datalist', { id: listId });
    ctx.editor.metadataFields(ctx.kbIds || []).then((fields) => {
      list.replaceChildren(...fields.map((m) => h('option', { value: m.name }, fieldLabel(m))));
    });
    return h('div', null,
      h('input', {
        id, type: 'text', value: value ?? '', list: listId, spellcheck: 'false', autocomplete: 'off',
        maxlength: f.inner.maxLength, placeholder: f.prop.placeholder || (f.nullable ? 'None' : 'e.g. date_added'),
        onInput: (e) => set(f.nullable && !e.target.value.trim() ? null : e.target.value.trim()),
      }),
      list);
  },

  model({ value, set, f, ctx, id, obj }) {
    const d = ctx.editor.modelDefaults;
    const listId = `${id}-models`;
    return h('div', null,
      h('input', {
        id, type: 'text', value: value ?? '', placeholder: placeholderFor(f, ctx, obj), list: listId, spellcheck: 'false',
        onInput: (e) => set(e.target.value.trim() || null),
      }),
      h('datalist', { id: listId }, [d.openai_model, d.anthropic_model].filter(Boolean).map((m) => h('option', { value: m }))));
  },

  textarea({ value, set, id }) {
    return autoGrow(h('textarea', { id, rows: 2, value: value ?? '', onInput: (e) => set(e.target.value) }));
  },

  prompt({ value, set, f, ctx, id }) {
    const ta = autoGrow(h('textarea', { id, rows: 3, class: 'prompt', value: value ?? '', spellcheck: 'false', onInput: (e) => set(e.target.value) }));
    return h('div', { class: 'prompt-box' }, ta, promptTools(ta, f, ctx, true));
  },

  template({ value, set, f, ctx, id }) {
    const input = h('input', { id, type: 'text', value: value ?? '', spellcheck: 'false', onInput: (e) => set(e.target.value) });
    return h('div', { class: 'input-group' }, input,
      iconButton('braces', 'Insert a variable', (e) => showVariables(e.currentTarget, input, ctx)));
  },

  value({ value, set, f, ctx, id }) {
    const input = h('input', {
      id, type: 'text', value: formatLoose(value ?? undefined), spellcheck: 'false', placeholder: f.prop.placeholder || 'e.g. 0, phones or {search@count}',
      onInput: (e) => set(e.target.value === '' ? null : parseLoose(e.target.value)),
    });
    return h('div', { class: 'input-group' }, input,
      iconButton('braces', 'Insert a variable', (e) => showVariables(e.currentTarget, input, ctx)));
  },

  number({ value, set, f, ctx, id, obj }) {
    const integer = f.inner.type === 'integer';
    const min = f.inner.minimum ?? f.inner.exclusiveMinimum;
    const max = f.inner.maximum ?? f.inner.exclusiveMaximum;
    return h('input', {
      id, type: 'number', value: value ?? '', min, max, step: integer ? '1' : 'any',
      placeholder: f.prop.placeholder || (f.nullable ? placeholderFor(f, ctx, obj) : f.prop.default !== undefined ? String(f.prop.default) : ''),
      onInput: (e) => {
        const text = e.target.value;
        if (text === '') return set(f.nullable ? null : f.prop.default);
        const n = Number(text);
        if (!Number.isNaN(n)) set(integer ? Math.trunc(n) : n);
      },
    });
  },

  bool({ value, set, id }) {
    return h('label', { class: 'switch-row' }, h('input', { id, type: 'checkbox', checked: !!value, onChange: (e) => set(e.target.checked) }), 'On');
  },

  enum({ value, set, f, ctx, id }) {
    const options = f.inner.enum.map((v) => h('option', { value: v },
      f.prop.labels?.[v] || (f.name === 'op' ? `${v}  (${OP_LABELS[v] || v})` : v)));
    if (f.nullable) options.unshift(h('option', { value: '' }, enumDefaultLabel(f, ctx)));
    const structural = f.name === 'provider'; // the model placeholder depends on it
    return h('select', { id, value: value ?? '', onChange: (e) => set(e.target.value === '' ? null : e.target.value, { structural }) }, options);
  },

  kbs({ value, set, ctx }) {
    let ids = Array.isArray(value) ? [...value] : [];
    const kbs = ctx.editor.kbs;
    const missing = ids.filter((id) => !kbs.some((k) => k.id === id));
    if (!kbs.length && !ids.length) {
      return h('div', { class: 'empty-note' }, 'No knowledge bases yet. Create one with POST /api/kbs, then add documents or a data source.');
    }
    const row = (kbId, label, note, cls = '') => h('label', { class: `check ${cls}` },
      h('input', {
        type: 'checkbox', checked: ids.includes(kbId),
        onChange: (e) => {
          ids = e.target.checked ? [...ids, kbId] : ids.filter((x) => x !== kbId);
          set(ids);
        },
      }),
      h('span', { class: 'check-label' }, label), note ? h('span', { class: 'muted' }, note) : null);
    return h('div', { class: 'checklist' },
      kbs.map((kb) => row(kb.id, kb.name, `${kb.document_count} docs`)),
      missing.map((kbId) => row(kbId, `#${kbId}`, 'does not exist', 'missing')));
  },

  node({ value, set, ctx, id }) {
    const nodes = ctx.editor.dsl.nodes;
    const ids = Object.keys(nodes).filter((n) => n !== ctx.nodeId && nodes[n].type !== 'begin');
    return h('select', { id, value: value ?? '', onChange: (e) => set(e.target.value || null) },
      h('option', { value: '' }, '— end the run —'),
      ids.map((n) => h('option', { value: n }, `${n} (${ctx.editor.types[nodes[n].type]?.title || nodes[n].type})`)),
      value && !ids.includes(value) ? h('option', { value }, `${value} (missing)`) : null);
  },

  kv({ value, set, f }) {
    const rows = Object.entries(value || {}).map(([k, v]) => ({ k, v: formatLoose(v) }));
    const box = h('div', { class: 'kv' });
    const emit = () => {
      const out = {};
      for (const r of rows) if (r.k.trim()) out[r.k.trim()] = r.v === '' ? '' : parseLoose(r.v);
      set(Object.keys(out).length || !f.nullable ? out : null);
    };
    const draw = () => {
      box.replaceChildren(
        ...rows.map((r, i) => h('div', { class: 'kv-row' },
          h('input', { type: 'text', placeholder: 'name', value: r.k, spellcheck: 'false', 'aria-label': 'Name', onInput: (e) => { r.k = e.target.value; emit(); } }),
          h('input', { type: 'text', placeholder: 'value', value: r.v, spellcheck: 'false', 'aria-label': 'Value', onInput: (e) => { r.v = e.target.value; emit(); } }),
          iconButton('x', 'Remove', () => { rows.splice(i, 1); emit(); draw(); }))),
        button('Add', {
          icon: 'plus', kind: 'small',
          onClick: () => { rows.push({ k: '', v: '' }); draw(); box.querySelectorAll('.kv-row input')[rows.length * 2 - 2]?.focus(); },
        }));
    };
    draw();
    return h('div', null, box, h('div', { class: 'help' }, 'Values are text; 42, true and null are typed values, "42" in quotes is text.'));
  },

  strings({ value, set }) {
    const items = Array.isArray(value) ? [...value] : [];
    const box = h('div', { class: 'string-list' });
    const draw = () => {
      box.replaceChildren(
        ...items.map((text, i) => h('div', { class: 'string-row' },
          h('input', { type: 'text', value: text, 'aria-label': `Item ${i + 1}`, onInput: (e) => { items[i] = e.target.value; set([...items]); } }),
          iconButton('x', 'Remove', () => { items.splice(i, 1); set([...items]); draw(); }))),
        button('Add', {
          icon: 'plus', kind: 'small',
          onClick: () => { items.push(''); set([...items]); draw(); box.querySelectorAll('input')[items.length - 1]?.focus(); },
        }));
    };
    draw();
    return box;
  },

  objects({ f, ctx, root, obj, path }) {
    const itemSchema = deref(f.inner.items, root);
    const list = Array.isArray(obj[f.name]) ? obj[f.name] : [];
    const min = f.inner.minItems || 0;
    const listChange = (key, fn, structural) => ctx.change(key, () => { obj[f.name] = list; fn(); }, structural);
    const label = singular(f.title);
    const cards = list.map((item, i) => {
      const title = item?.name || filterTitle(item) || `${label[0].toUpperCase()}${label.slice(1)} ${i + 1}`;
      const move = (to) => listChange(null, () => { list.splice(to, 0, list.splice(i, 1)[0]); }, true);
      return h('div', { class: 'card' },
        h('div', { class: 'card-head' },
          h('span', { class: 'card-title' }, title),
          iconButton('chevronUp', 'Move up', () => move(i - 1), { disabled: i === 0 }),
          iconButton('chevronDown', 'Move down', () => move(i + 1), { disabled: i === list.length - 1 }),
          iconButton('trash', `Remove this ${label}`, () => listChange(null, () => { list.splice(i, 1); }, true), { disabled: list.length <= min })),
        h('div', { class: 'card-body' }, renderFields(item, itemSchema, root, { ...ctx, change: listChange }, `${path}.${f.name}.${i}`)));
    });
    const add = button(`Add ${label}`, { icon: 'plus', kind: 'small', onClick: () => listChange(null, () => { list.push(newItem(itemSchema, root, f, list, ctx)); }, true) });
    const fill = f.prop.fill_from_kbs ? button('Add from the knowledge bases', {
      icon: 'download', kind: 'small', title: 'Add the metadata fields found in the selected knowledge bases',
      onClick: async () => {
        const fields = await ctx.editor.metadataFields(ctx.kbIds || []);
        const known = new Set(list.map((x) => x?.name));
        const fresh = fields.filter((m) => !known.has(m.name));
        if (!fields.length) return toast('No metadata found in the selected knowledge bases', 'error');
        if (!fresh.length) return toast('Every field is in the list already');
        listChange(null, () => {
          for (const m of fresh) {
            const examples = m.examples.length ? `e.g. ${m.examples.map((x) => String(x)).join(', ')}` : '';
            list.push({ name: m.name, ...(m.type !== 'text' ? { type: m.type } : {}), ...(examples ? { description: examples } : {}) });
          }
        }, true);
        toast(`Added ${fresh.length} field${fresh.length > 1 ? 's' : ''}. Describe them for the model, and remove the ones it shouldn't use.`);
      },
    }) : null;
    return h('div', { class: 'cards' }, cards, h('div', { class: 'card-actions' }, add, fill));
  },
};

function fieldLabel(m) {
  const examples = m.examples.length ? ` · e.g. ${m.examples.map((x) => String(x)).join(', ')}` : '';
  return `${m.type}${examples}`.slice(0, 90);
}

// "price < 500" as the title of a filter card.
function filterTitle(item) {
  if (!item || !item.field || !item.op) return '';
  const value = item.value === undefined || item.value === null ? '…' : formatLoose(item.value);
  return `${item.field} ${item.op} ${value}`.slice(0, 60);
}

// --- Variables and big editor ------------------------------------------------------------

function promptTools(field, f, ctx, withExpand) {
  return h('div', { class: 'prompt-tools' },
    f.prop.variables !== false ? button('Variable', { icon: 'braces', kind: 'small', title: 'Insert a variable', onClick: (e) => showVariables(e.currentTarget, field, ctx) }) : null,
    (f.prop.placeholders || []).map((p) => button(p, { kind: 'chip', title: `Insert ${p}`, onClick: () => insertAtCursor(field, p.replace('NAME', '')) })),
    h('span', { class: 'spacer' }),
    withExpand ? iconButton('expand', 'Open in a large editor', () => bigEditor(field, f, ctx)) : null);
}

export function showVariables(anchor, field, ctx) {
  const groups = variablesFor(ctx.editor.dsl, ctx.nodeId, ctx.editor.types, ctx.editor.sysVars);
  const content = h('div', { class: 'var-menu' }, groups.map((g) => h('div', { class: 'var-group' },
    h('div', { class: 'var-group-title' }, g.title),
    g.items.map((item) => h('button', {
      type: 'button', class: 'var-item',
      onClick: () => { closePopover(); insertAtCursor(field, item.value); },
    }, h('code', null, item.value), h('span', null, item.description))))));
  popover(anchor, content, { className: 'var-popover' });
}

function bigEditor(small, f, ctx) {
  const big = h('textarea', { class: 'prompt big', value: small.value, spellcheck: 'false' });
  big.addEventListener('input', () => {
    small.value = big.value;
    small.dispatchEvent(new Event('input', { bubbles: true }));
  });
  modal({
    title: f.title,
    wide: true,
    body: h('div', { class: 'big-editor' }, big, promptTools(big, f, ctx, false), f.prop.description ? h('div', { class: 'help' }, f.prop.description) : null),
    actions: [{ label: 'Done', kind: 'primary' }],
  });
}

// --- Panels ---------------------------------------------------------------------------------

function section(title, ...children) {
  return h('div', { class: 'section' }, title ? h('div', { class: 'section-title' }, title) : null, children);
}

// The editor refreshes this slot after each validation (updatePanelProblems).
function errorList(messages) {
  return h('div', { class: 'problems-slot' }, (messages || []).map((m) => h('div', { class: 'problem' }, icon('alert'), h('span', null, m))));
}

export function renderNodePanel(editor, id) {
  const node = editor.dsl.nodes[id];
  const info = editor.types[node.type] || { title: node.type, schema: { properties: {} } };
  const ctx = {
    editor,
    nodeId: id,
    change: (key, fn, structural) => editor.change(key, fn, { source: structural ? 'panel' : 'form' }),
  };

  const idInput = h('input', { type: 'text', id: 'f-node-id', value: id, spellcheck: 'false', 'aria-label': 'Node id' });
  const applyRename = () => {
    const value = idInput.value.trim();
    if (value === id) return;
    if (!NODE_ID.test(value)) {
      toast('Node ids start with a letter and contain letters, digits, _ and - only', 'error');
      idInput.value = id;
    } else if (editor.dsl.nodes[value]) {
      toast(`There is already a node called ${value}`, 'error');
      idInput.value = id;
    } else editor.renameNode(id, value);
  };
  idInput.addEventListener('change', applyRename);
  idInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') idInput.blur();
    if (e.key === 'Escape') { idInput.value = id; idInput.blur(); }
  });

  let next = null;
  if (!ROUTERS.has(node.type)) {
    const nodes = editor.dsl.nodes;
    const targets = Object.keys(nodes).filter((n) => n !== id && nodes[n].type !== 'begin');
    next = h('div', { class: 'field' },
      h('div', { class: 'field-label' }, h('label', null, 'Next node')),
      h('select', { value: node.next ?? '', onChange: (e) => editor.connect(id, 'next', e.target.value || null) },
        h('option', { value: '' }, '— end the run —'),
        targets.map((n) => h('option', { value: n }, `${n} (${editor.types[nodes[n].type]?.title || nodes[n].type})`)),
        node.next && !targets.includes(node.next) ? h('option', { value: node.next }, `${node.next} (missing)`) : null),
      h('div', { class: 'help' }, 'Or drag from the dot on the right of the node to another node.'));
  }

  return h('div', { class: 'panel' },
    h('div', { class: `panel-head t-${node.type}` },
      h('span', { class: 'node-icon big' }, icon(node.type)),
      h('div', null, h('div', { class: 'panel-type' }, info.title), h('div', { class: 'panel-desc' }, info.description))),
    errorList(editor.errorsByNode[id]),
    section(null,
      h('div', { class: 'field' },
        h('div', { class: 'field-label' }, h('label', null, 'Node id')),
        idInput,
        h('div', { class: 'help' }, 'Other nodes use its outputs as {id@output}. Renaming updates them.')),
      next),
    section('Settings', renderFields(node.params, info.schema, info.schema, ctx, 'params')),
    section('Outputs', h('div', { class: 'outputs-slot' }, outputList(editor, id))),
    h('div', { class: 'panel-actions' },
      button('Duplicate', { icon: 'copy', onClick: () => editor.duplicateNode(id) }),
      node.type === 'begin' ? null : button('Delete', { icon: 'trash', kind: 'danger-ghost', onClick: () => editor.deleteNode(id) })));
}

// The {id@output} variables a node gives later nodes. The editor refreshes it as you type (begin inputs).
export function outputList(editor, id) {
  const node = editor.dsl.nodes[id];
  const outputs = node ? nodeOutputs(node, editor.types[node.type]) : [];
  if (!outputs.length) {
    return h('div', { class: 'muted' }, node?.type === 'begin' ? 'Add inputs above to use them as variables.' : 'None');
  }
  return h('div', { class: 'outputs' }, outputs.map((o) => h('button', {
    type: 'button', class: 'output', title: 'Copy',
    onClick: () => copy(`{${id}@${o.name}}`),
  }, h('code', null, `{${id}@${o.name}}`), h('span', null, o.description))));
}

export function renderEdgePanel(editor, sel) {
  const node = editor.dsl.nodes[sel.node];
  const port = outPorts(node).find((p) => p.key === sel.port);
  const from = port?.label ? `${sel.node} (${port.label})` : sel.node;
  return h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('div', null, h('div', { class: 'panel-type' }, 'Connection'),
      h('div', { class: 'panel-desc' }, `From ${from} to ${port?.target || 'nothing'}.`))),
    h('div', { class: 'panel-actions' },
      button('Remove connection', { icon: 'trash', kind: 'danger-ghost', onClick: () => editor.deleteSelection() }),
      button(`Open ${sel.node}`, { onClick: () => editor.select({ type: 'node', id: sel.node }) })));
}

export function renderWorkflowPanel(editor) {
  const wf = editor.workflow;
  const nodes = Object.keys(editor.dsl.nodes).length;
  const apiUrl = `${location.origin}${editor.apiRoot}workflows/${wf.id}/run`;
  const snippet = `curl -N ${apiUrl} \\\n  -H "Authorization: Bearer $API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query": "Hello", "user_id": "user-1"}'`;
  const maxSteps = editor.dsl.max_steps ?? 50;
  return h('div', { class: 'panel' },
    h('div', { class: 'panel-head' },
      h('span', { class: 'node-icon big' }, icon('workflow')),
      h('div', null, h('div', { class: 'panel-type' }, 'Workflow'), h('div', { class: 'panel-desc' }, 'Select a node to edit it. These settings apply to the whole workflow.'))),
    errorList(editor.generalErrors),
    section(null,
      h('div', { class: 'field' },
        h('div', { class: 'field-label' }, h('label', { for: 'f-wf-name' }, 'Name')),
        h('input', { type: 'text', id: 'f-wf-name', maxlength: 200, value: editor.name, onInput: (e) => editor.change('wf:name', () => { editor.name = e.target.value; }, { source: 'form' }) })),
      h('div', { class: 'field' },
        h('div', { class: 'field-label' }, h('label', null, 'Description')),
        autoGrow(h('textarea', { rows: 2, id: 'f-wf-description', value: editor.description, onInput: (e) => editor.change('wf:description', () => { editor.description = e.target.value; }, { source: 'form' }) }))),
      h('div', { class: 'field' },
        h('div', { class: 'field-label' }, h('label', null, 'Max steps')),
        h('input', {
          type: 'number', min: 5, max: 500, step: 1, value: maxSteps,
          onInput: (e) => editor.change('wf:max_steps', () => {
            const n = Math.trunc(Number(e.target.value));
            if (!e.target.value || n === 50) delete editor.dsl.max_steps; else if (!Number.isNaN(n)) editor.dsl.max_steps = n;
          }, { source: 'form' }),
        }),
        h('div', { class: 'help' }, 'A run stops after this many node runs (5 to 500), in case a loop never ends.'))),
    section('Use it from your app',
      h('div', { class: 'help' }, `Workflow id ${wf.id}. Saved changes are live right away. Streams Server-Sent Events; add "stream": false for one JSON answer, and "conversation_id" to continue a conversation.`),
      h('pre', { class: 'snippet' }, snippet),
      button('Copy', { icon: 'copy', kind: 'small', onClick: () => copy(snippet) })),
    section(null, h('div', { class: 'muted' }, `${nodes} nodes · last saved ${new Date(wf.updated_at).toLocaleString()}`)),
    h('div', { class: 'panel-actions' },
      button('Export', { icon: 'download', onClick: () => editor.exportJson() }),
      button('Edit JSON', { icon: 'code', onClick: () => editor.editJson() }),
      button('Delete workflow', { icon: 'trash', kind: 'danger-ghost', onClick: () => editor.deleteWorkflow() })));
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`Copied ${text.length > 40 ? 'to the clipboard' : text}`);
  } catch {
    toast('Could not copy (the clipboard needs HTTPS or localhost)', 'error');
  }
}

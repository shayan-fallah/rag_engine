// Pure helpers over the workflow DSL: ports, connections, renames, layout, variables.

export const ROUTERS = new Set(['categorize', 'switch']);
export const NODE_ID = /^[A-Za-z][A-Za-z0-9_-]*$/;

// The output ports of a node: [{key, label, target}]. key says where the target is stored.
export function outPorts(node) {
  const p = node.params || {};
  if (node.type === 'categorize') {
    return (p.categories || []).map((c, i) => ({ key: `categories.${i}`, label: c.name || `category ${i + 1}`, target: c.next ?? null }));
  }
  if (node.type === 'switch') {
    return [
      ...(p.cases || []).map((c, i) => ({ key: `cases.${i}`, label: caseLabel(c, i), target: c.next ?? null })),
      { key: 'default', label: 'else', target: p.default ?? null },
    ];
  }
  return [{ key: 'next', label: '', target: node.next ?? null }];
}

export function caseLabel(c, i) {
  const conds = c.conditions || [];
  if (!conds.length) return `case ${i + 1}`;
  const text = conds.map((x) => {
    const value = ['empty', 'not_empty'].includes(x.op) ? '' : ` ${x.value === undefined || x.value === null ? '' : typeof x.value === 'string' ? x.value : JSON.stringify(x.value)}`;
    return `${x.var || '?'} ${(x.op || '').replace('_', ' ')}${value}`;
  }).join(c.logic === 'or' ? ' or ' : ' and ');
  return `if ${text}`;
}

export function setTarget(node, key, target) {
  target = target || null;
  if (key === 'next') {
    if (target) node.next = target; else delete node.next;
    return;
  }
  node.params = node.params || {};
  if (key === 'default') {
    if (target) node.params.default = target; else delete node.params.default;
    return;
  }
  const [list, index] = key.split('.');
  const item = node.params[list]?.[Number(index)];
  if (!item) return;
  if (target) item.next = target; else delete item.next;
}

export function clearTargets(node) {
  for (const port of outPorts(node)) setTarget(node, port.key, null);
}

export function newNodeId(dsl, base) {
  base = base.replace(/[^A-Za-z0-9_-]/g, '_');
  if (!/^[A-Za-z]/.test(base)) base = `n_${base}`;
  for (let i = 1; ; i++) {
    const id = `${base}_${i}`;
    if (!dsl.nodes[id]) return id;
  }
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');

function rewriteStrings(value, fn) {
  if (typeof value === 'string') return fn(value);
  if (Array.isArray(value)) return value.map((v) => rewriteStrings(v, fn));
  if (value && typeof value === 'object') {
    for (const k of Object.keys(value)) value[k] = rewriteStrings(value[k], fn);
  }
  return value;
}

// Rename a node: its key, every connection to it and every {old@output} variable.
export function renameNode(dsl, oldId, newId) {
  const entries = Object.entries(dsl.nodes).map(([id, node]) => [id === oldId ? newId : id, node]);
  dsl.nodes = Object.fromEntries(entries);
  if (dsl.layout?.[oldId]) {
    dsl.layout[newId] = dsl.layout[oldId];
    delete dsl.layout[oldId];
  }
  const pattern = new RegExp(`\\{${escapeRe(oldId)}@`, 'g');
  for (const node of Object.values(dsl.nodes)) {
    for (const port of outPorts(node)) if (port.target === oldId) setTarget(node, port.key, newId);
    if (node.params) rewriteStrings(node.params, (s) => s.replace(pattern, `{${newId}@`));
  }
}

export function removeNode(dsl, id) {
  delete dsl.nodes[id];
  if (dsl.layout) delete dsl.layout[id];
  for (const node of Object.values(dsl.nodes)) {
    for (const port of outPorts(node)) if (port.target === id) setTarget(node, port.key, null);
  }
}

export function edges(dsl) {
  const list = [];
  for (const [id, node] of Object.entries(dsl.nodes)) {
    for (const port of outPorts(node)) {
      if (port.target && dsl.nodes[port.target]) list.push({ from: id, port: port.key, to: port.target });
    }
  }
  return list;
}

// Nodes that can run before `id` (they can reach it).
export function ancestors(dsl, id) {
  const parents = {};
  for (const e of edges(dsl)) (parents[e.to] = parents[e.to] || []).push(e.from);
  const seen = new Set();
  const stack = [...(parents[id] || [])];
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    stack.push(...(parents[n] || []));
  }
  seen.delete(id);
  return seen;
}

export function beginId(dsl) {
  return Object.keys(dsl.nodes).find((id) => dsl.nodes[id].type === 'begin');
}

// Columns by distance from begin; nodes nothing leads to go in a last column.
export function autoLayout(dsl, sizes = {}) {
  const start = beginId(dsl);
  const depth = {};
  const order = [];
  if (start) {
    depth[start] = 0;
    const queue = [start];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const port of outPorts(dsl.nodes[id])) {
        const t = port.target;
        if (t && dsl.nodes[t] && depth[t] === undefined) {
          depth[t] = depth[id] + 1;
          queue.push(t);
        }
      }
    }
  }
  const maxDepth = Math.max(0, ...Object.values(depth));
  for (const id of Object.keys(dsl.nodes)) {
    if (depth[id] === undefined) { depth[id] = maxDepth + 1; order.push(id); }
  }
  const columns = {};
  for (const id of order) (columns[depth[id]] = columns[depth[id]] || []).push(id);
  const layout = {};
  const gapX = 300, gapY = 40;
  for (const [col, ids] of Object.entries(columns)) {
    const heights = ids.map((id) => sizes[id]?.h || 110);
    const total = heights.reduce((a, b) => a + b, 0) + gapY * (ids.length - 1);
    let y = -total / 2;
    ids.forEach((id, i) => {
      layout[id] = { x: Number(col) * gapX, y: Math.round(y) };
      y += heights[i] + gapY;
    });
  }
  return layout;
}

// The outputs ({id@name}) a node gives later nodes.
export function nodeOutputs(node, info) {
  if (node.type === 'begin') return Object.keys(node.params?.inputs || {}).map((name) => ({ name, description: 'Input' }));
  return Object.entries(info?.outputs || {}).map(([name, description]) => ({ name, description }));
}

// Variables a node can use, grouped: {sys.*} and the outputs of the nodes before it.
export function variablesFor(dsl, id, types, sysVars) {
  const groups = [{
    title: 'System',
    items: Object.entries(sysVars).map(([name, description]) => ({ value: `{sys.${name}}`, description })),
  }];
  let before = ancestors(dsl, id);
  if (!before.size) before = new Set(Object.keys(dsl.nodes).filter((n) => n !== id)); // not connected yet
  for (const nodeId of Object.keys(dsl.nodes)) {
    if (!before.has(nodeId)) continue;
    const node = dsl.nodes[nodeId];
    const outputs = nodeOutputs(node, types[node.type]);
    if (!outputs.length) continue;
    groups.push({
      title: `${nodeId} · ${types[node.type]?.title || node.type}`,
      items: outputs.map((o) => ({ value: `{${nodeId}@${o.name}}`, description: o.description })),
    });
  }
  return groups;
}

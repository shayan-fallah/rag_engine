// Studio entry point: loads node types and knowledge bases, then routes
//   #/        the workflow list
//   #/w/<id>  the editor

import { api } from './api.js';
import { EditorPage } from './editor.js';
import { ListPage } from './list.js';
import { confirmDialog, h } from './ui.js';

const root = document.getElementById('app');
const app = { current: null, types: {}, sysVars: {}, modelDefaults: {}, kbs: [] };
let currentHash = null;

async function boot() {
  try {
    const [meta, kbs] = await Promise.all([api.get('node-types'), api.get('kbs')]);
    app.types = Object.fromEntries(meta.types.map((t) => [t.type, t]));
    app.sysVars = meta.sys_vars;
    app.modelDefaults = meta.model_defaults;
    app.kbs = kbs;
  } catch (err) {
    root.replaceChildren(h('div', { class: 'boot error' }, h('h1', null, 'Could not load the studio'), h('p', null, err.message)));
    return;
  }
  window.addEventListener('hashchange', route);
  window.addEventListener('beforeunload', (e) => {
    app.current?.commitEdits?.();
    if (app.current?.dirty) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
  route();
}

async function route() {
  const target = location.hash || '#/';
  if (target === currentHash) return;
  app.current?.commitEdits?.();
  if (app.current?.dirty) {
    history.pushState(null, '', currentHash); // stay until the user decides (Back still works after)
    const leave = await confirmDialog('Leave without saving?', 'This workflow has unsaved changes. They will be lost.', { confirm: 'Leave', danger: true });
    if (!leave) return;
    app.current.dirty = false;
    history.replaceState(null, '', target);
  }
  currentHash = target;
  app.current?.destroy?.();
  const match = target.match(/^#\/w\/(\d+)$/);
  app.current = match ? new EditorPage(root, app, Number(match[1])) : new ListPage(root, app);
}

boot();

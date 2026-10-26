// Small DOM helpers, icons, dialogs and toasts. No framework: h() builds elements.

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  let value;
  for (const [key, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (key === 'class') el.className = v;
    else if (key === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (key === 'dataset') Object.assign(el.dataset, v);
    else if (key.startsWith('on') && typeof v === 'function') el.addEventListener(key.slice(2).toLowerCase(), v);
    else if (key === 'value') value = v;
    else if (key === 'checked' || key === 'disabled' || key === 'selected' || key === 'open') el[key] = !!v;
    else el.setAttribute(key, v === true ? '' : v);
  }
  append(el, children);
  if (value !== undefined) el.value = value; // after <option>s exist
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  return el;
}

const ICONS = {
  begin: '<path d="M7 4.5v15l12-7.5z"/>',
  retrieval: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  llm: '<path d="M12 3.5l1.8 4.7 4.7 1.8-4.7 1.8L12 16.5l-1.8-4.7L5.5 10l4.7-1.8z"/><path d="M18.5 15.5l.8 1.9 1.9.8-1.9.8-.8 1.9-.8-1.9-1.9-.8 1.9-.8z"/>',
  agent: '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 8V4.5"/><circle cx="12" cy="3.5" r="1"/><path d="M9 13.5v1.5M15 13.5v1.5"/>',
  categorize: '<path d="M3.5 12H9"/><path d="M9 12l6-6.5h5.5"/><path d="M9 12h11.5"/><path d="M9 12l6 6.5h5.5"/>',
  switch: '<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="7" r="2"/><path d="M6 7v10"/><path d="M18 9c0 5-12 3-12 8"/>',
  message: '<path d="M4 5.5h16v11H9.5L5 20v-3.5H4z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 9V5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v8A1.5 1.5 0 0 0 5.5 15H9"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/>',
  save: '<path d="M5 4h11l4 4v12H5z"/><path d="M8 4v5h7"/><path d="M8 20v-6h8v6"/>',
  play: '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>',
  zoomIn: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M11 8v6M8 11h6"/>',
  zoomOut: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M8 11h6"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  layout: '<rect x="3" y="4" width="7" height="6" rx="1.5"/><rect x="14" y="14" width="7" height="6" rx="1.5"/><path d="M10 7h3a2 2 0 0 1 2 2v5"/>',
  chevronDown: '<path d="m6 9 6 6 6-6"/>',
  chevronUp: '<path d="m6 15 6-6 6 6"/>',
  chevronRight: '<path d="m9 6 6 6-6 6"/>',
  braces: '<path d="M8 4H7a2 2 0 0 0-2 2v4l-2 2 2 2v4a2 2 0 0 0 2 2h1M16 4h1a2 2 0 0 1 2 2v4l2 2-2 2v4a2 2 0 0 1-2 2h-1"/>',
  back: '<path d="M19 12H5M11 18l-6-6 6-6"/>',
  download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
  upload: '<path d="M12 16V5M7 9.5l5-5 5 5M5 20h14"/>',
  code: '<path d="m8 8-4 4 4 4M16 8l4 4-4 4"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5h.01"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="1.5"/>',
  send: '<path d="M4 12 20 4l-5 16-3.5-6.5z"/><path d="M11.5 13.5 20 4"/>',
  refresh: '<path d="M4.5 12a7.5 7.5 0 0 1 13-5.1L20 9.5M20 4.5v5h-5"/><path d="M19.5 12a7.5 7.5 0 0 1-13 5.1L4 14.5M4 19.5v-5h5"/>',
  logo: '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="12" r="2.5"/><circle cx="6" cy="18" r="2.5"/><path d="M8.5 6H12a3 3 0 0 1 3 3v.5M8.5 18H12a3 3 0 0 0 3-3v-.5"/>',
  expand: '<path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
  workflow: '<rect x="3" y="3" width="7" height="6" rx="1.5"/><rect x="14" y="15" width="7" height="6" rx="1.5"/><path d="M6.5 9v3.5a2.5 2.5 0 0 0 2.5 2.5h5"/>',
  logout: '<path d="M14 4h5v16h-5M10 8l-4 4 4 4M6 12h10"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/>',
  log: '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4.5 6h.01M4.5 12h.01M4.5 18h.01"/>',
};

export function icon(name, cls = '') {
  const el = svg('svg', {
    viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2',
    'stroke-linecap': 'round', 'stroke-linejoin': 'round', class: `icon ${cls}`, 'aria-hidden': 'true',
  });
  el.innerHTML = ICONS[name] || ICONS.alert; // static markup from this file only
  return el;
}

export function button(label, props = {}) {
  const { icon: iconName, kind = '', title, ...rest } = props;
  return h('button', { type: 'button', class: `btn ${kind}`.trim(), title: title || (label ? null : rest['aria-label']), ...rest },
    iconName ? icon(iconName) : null, label ? h('span', null, label) : null);
}

export function iconButton(iconName, title, onClick, extra = {}) {
  return h('button', { type: 'button', class: `icon-btn ${extra.class || ''}`.trim(), title, 'aria-label': title, onClick, disabled: extra.disabled }, icon(iconName));
}

// --- Dialogs ----------------------------------------------------------------------

export function modal({ title, body, actions = [], wide = false, onClose }) {
  const backdrop = h('div', { class: 'modal-backdrop' });
  const close = (value) => {
    closePopover();
    backdrop.remove();
    document.removeEventListener('keydown', onKey, true);
    onClose?.(value);
  };
  const onKey = (e) => {
    if (e.key !== 'Escape') return;
    e.stopPropagation();
    if (openPopover) closePopover(); // e.g. the variable menu inside the large editor
    else close(undefined);
  };
  const footer = h('div', { class: 'modal-actions' },
    actions.map((a) => button(a.label, {
      kind: a.kind || '', icon: a.icon, disabled: a.disabled,
      onClick: async (e) => {
        if (!a.onClick) return close(a.value);
        const btn = e.currentTarget;
        btn.disabled = true; // no double submit while it runs
        try {
          const result = await a.onClick();
          if (result !== false) close(result); // false: keep the dialog open
        } finally {
          btn.disabled = false;
        }
      },
    })));
  const dialog = h('div', { class: `modal ${wide ? 'wide' : ''}`, role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
    h('div', { class: 'modal-head' }, h('h2', null, title), iconButton('x', 'Close', () => close(undefined))),
    h('div', { class: 'modal-body' }, body),
    actions.length ? footer : null);
  backdrop.append(dialog);
  backdrop.addEventListener('pointerdown', (e) => { if (e.target === backdrop) close(undefined); });
  document.addEventListener('keydown', onKey, true);
  document.body.append(backdrop);
  const focusable = dialog.querySelector('.modal-body input, .modal-body textarea, .modal-body select');
  (focusable || dialog.querySelector('.modal-actions .btn.primary'))?.focus();
  return { close, dialog, footer };
}

export function confirmDialog(title, message, { confirm = 'OK', danger = false } = {}) {
  return new Promise((resolve) => {
    modal({
      title,
      body: h('p', { class: 'dialog-text' }, message),
      actions: [
        { label: 'Cancel', value: false },
        { label: confirm, kind: danger ? 'danger' : 'primary', value: true },
      ],
      onClose: (value) => resolve(!!value),
    });
  });
}

// --- Toasts -----------------------------------------------------------------------

let toastBox;
export function toast(message, kind = 'info', ms = 3500) {
  if (!toastBox) {
    toastBox = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(toastBox);
  }
  const el = h('div', { class: `toast ${kind}` }, icon(kind === 'error' ? 'alert' : 'check'), h('span', null, message));
  toastBox.append(el);
  setTimeout(() => { el.classList.add('leaving'); setTimeout(() => el.remove(), 250); }, ms);
}

// --- Popover menus ------------------------------------------------------------------

let openPopover = null;
export function closePopover() {
  if (openPopover) { openPopover.cleanup(); openPopover = null; }
}

// Shows `content` in a floating box next to `anchor` (an element, or {x, y} in page coordinates).
export function popover(anchor, content, { className = '', align = 'start' } = {}) {
  closePopover();
  const box = h('div', { class: `popover ${className}` }, content);
  document.body.append(box);
  const r = anchor instanceof Element ? anchor.getBoundingClientRect() : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
  const bw = box.offsetWidth, bh = box.offsetHeight;
  let left = align === 'end' ? r.right - bw : r.left;
  let top = r.bottom + 6;
  if (top + bh > window.innerHeight - 8) top = Math.max(8, r.top - bh - 6);
  left = Math.max(8, Math.min(left, window.innerWidth - bw - 8));
  box.style.left = `${left}px`;
  box.style.top = `${top}px`;
  const onDown = (e) => { if (!box.contains(e.target) && !(anchor instanceof Element && anchor.contains(e.target))) closePopover(); };
  const onKey = (e) => { if (e.key === 'Escape') closePopover(); };
  setTimeout(() => document.addEventListener('pointerdown', onDown, true));
  document.addEventListener('keydown', onKey);
  openPopover = {
    cleanup() {
      box.remove();
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
    },
  };
  return box;
}

// --- Misc -----------------------------------------------------------------------------

export const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));

export function deepEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function timeAgo(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)} days ago`;
  return new Date(iso).toLocaleDateString();
}

export function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' });
  const a = h('a', { href: URL.createObjectURL(blob), download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function slug(text) {
  return (text || 'workflow').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'workflow';
}

// Grow a textarea with its content, up to max px.
export function autoGrow(textarea, max = 420) {
  const fit = () => {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight + 2, max)}px`;
  };
  textarea.addEventListener('input', fit);
  requestAnimationFrame(fit);
  return textarea;
}

// Insert text at the caret of an input/textarea and fire "input".
export function insertAtCursor(field, text) {
  if (!field.isConnected) return; // the panel was redrawn (e.g. undo): this field is gone
  const start = field.selectionStart ?? field.value.length;
  const end = field.selectionEnd ?? field.value.length;
  field.value = field.value.slice(0, start) + text + field.value.slice(end);
  field.focus();
  field.selectionStart = field.selectionEnd = start + text.length;
  field.dispatchEvent(new Event('input', { bubbles: true }));
}

// Values typed in a form: JSON when it parses (numbers, true, null, "quoted text"), else plain text.
export function parseLoose(text) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function formatLoose(value) {
  if (value === undefined) return '';
  if (typeof value === 'string') {
    try {
      JSON.parse(value);
      return JSON.stringify(value); // "123" stays text: show the quotes
    } catch {
      return value;
    }
  }
  return JSON.stringify(value);
}

// The REST API, called with the logged-in superuser's session (+ CSRF token for writes).

export const config = JSON.parse(document.getElementById('studio-config').textContent);

export class ApiError extends Error {
  constructor(status, body) {
    super(describe(status, body));
    this.status = status;
    this.body = body;
    this.errors = (body && body.errors) || []; // [{node, message}] for invalid workflows
  }
}

function describe(status, body) {
  if (status === 401) return 'Your session has ended. Reload the page to log in again.';
  if (!body) return `Request failed (${status})`;
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((d) => `${(d.loc || []).slice(1).join('.') || 'request'}: ${d.msg}`).join('; ');
  }
  return `Request failed (${status})`;
}

function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

async function call(method, path, body, accept = 'application/json', signal) {
  const headers = { Accept: accept };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (method !== 'GET') headers['X-CSRFToken'] = csrfToken();
  const response = await fetch(config.apiRoot + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'same-origin',
    signal,
  });
  if (!response.ok) {
    let data = null;
    try { data = await response.json(); } catch { /* not JSON */ }
    throw new ApiError(response.status, data);
  }
  return response;
}

async function json(method, path, body) {
  const response = await call(method, path, body);
  return response.status === 204 ? null : response.json();
}

export const api = {
  get: (path) => json('GET', path),
  post: (path, body) => json('POST', path, body ?? {}),
  put: (path, body) => json('PUT', path, body),
  del: (path) => json('DELETE', path),
};

// POST and read Server-Sent Events: onEvent(event) for each `data: {...}` line.
export async function stream(path, body, onEvent, signal) {
  const response = await call('POST', path, body, 'text/event-stream', signal);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) throw new Error('The connection closed before the run finished. Check the server log.');
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of block.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') return;
        onEvent(JSON.parse(data));
      }
    }
  }
}

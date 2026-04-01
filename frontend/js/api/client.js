export const resolveApiBase = () => {
  const seeded = window.PID_API_BASE;
  if (seeded && typeof seeded === 'string') return seeded.replace(/\/$/, '');
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:3443`;
};

export const apiUrl = (path) => `${resolveApiBase()}${path.startsWith('/') ? path : `/${path}`}`;

const mergeAbortSignals = (signals) => {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  (signals || []).filter(Boolean).forEach((signal) => {
    if (signal.aborted) {
      controller.abort();
      return;
    }
    signal.addEventListener('abort', onAbort, { once: true });
  });
  return controller.signal;
};

export const apiFetch = async (path, options = {}) => {
  const { timeoutMs, signal, ...rest } = options || {};
  const timeout = Number(timeoutMs) || 0;
  if (timeout <= 0) {
    return fetch(apiUrl(path), { ...rest, signal });
  }

  const timeoutController = new AbortController();
  const mergedSignal = mergeAbortSignals([signal, timeoutController.signal]);
  const timer = window.setTimeout(() => timeoutController.abort(), timeout);
  try {
    return await fetch(apiUrl(path), { ...rest, signal: mergedSignal });
  } finally {
    window.clearTimeout(timer);
  }
};

export const apiGetJson = async (path) => {
  const res = await apiFetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const apiPostJson = async (path, payload) => {
  const res = await apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {})
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

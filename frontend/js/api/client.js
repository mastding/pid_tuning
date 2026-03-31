export const resolveApiBase = () => {
  const seeded = window.PID_API_BASE;
  if (seeded && typeof seeded === 'string') return seeded.replace(/\/$/, '');
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:3443`;
};

export const apiUrl = (path) => `${resolveApiBase()}${path.startsWith('/') ? path : `/${path}`}`;

export const apiFetch = async (path, options = {}) => {
  return fetch(apiUrl(path), options);
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


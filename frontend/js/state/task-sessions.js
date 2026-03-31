import { apiFetch, apiGetJson } from '../api/client.js';

export const loadTaskSessionsPayload = async (storageKey) => {
  let payload = null;

  try {
    const remote = await apiGetJson('/api/task-sessions');
    if (remote && Array.isArray(remote.items)) {
      payload = remote;
      localStorage.setItem(storageKey, JSON.stringify(remote));
      return payload;
    }
  } catch (_) {}

  const raw = localStorage.getItem(storageKey);
  if (!raw || raw === '{}') return null;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch (_) {
    return null;
  }
};

export const saveTaskSessionsPayload = async (storageKey, payload) => {
  localStorage.setItem(storageKey, JSON.stringify(payload));
  try {
    await apiFetch('/api/task-sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (_) {}
};


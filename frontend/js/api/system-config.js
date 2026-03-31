import { apiFetch, apiGetJson, apiPostJson } from './client.js';

export const fetchSystemConfig = async () => {
  return apiGetJson('/api/system-config');
};

export const saveSystemConfig = async (payload) => {
  const res = await apiFetch('/api/system-config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {})
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const testModelConnectivity = async (modelPayload) => {
  return apiPostJson('/api/system-config/test-model', modelPayload ?? {});
};


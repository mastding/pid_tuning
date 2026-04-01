import { apiFetch, apiPostJson } from './client.js';

export const fetchPidChartData = async (payload) => {
  return apiPostJson('/api/tuning/pid-chart-data', payload ?? {});
};

export const fetchPidPredictionCurve = async (payload) => {
  return apiPostJson('/api/tuning/pid-prediction-curve', payload ?? {});
};

export const startTuneStream = async (formData, options = {}) => {
  const { timeoutMs, signal } = options || {};
  const res = await apiFetch('/api/tune_stream', {
    method: 'POST',
    body: formData,
    timeoutMs: Number(timeoutMs) || 15000,
    signal
  });
  if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
  return res;
};

export const inspectCsvWindows = async (formData) => {
  const res = await apiFetch('/api/tuning/csv/inspect-windows', {
    method: 'POST',
    body: formData
  });
  const payload = await res.json().catch(() => null);
  if (!res.ok) {
    const message = payload?.message || `HTTP error! status: ${res.status}`;
    throw new Error(message);
  }
  if (!payload || payload.code !== 0) {
    throw new Error(payload?.message || '窗口识别失败');
  }
  return payload.data || {};
};

export const inspectCsvLoops = async (formData) => {
  const res = await apiFetch('/api/tuning/csv/inspect-loops', {
    method: 'POST',
    body: formData
  });
  const payload = await res.json().catch(() => null);
  if (!res.ok) {
    const message = payload?.message || `HTTP error! status: ${res.status}`;
    throw new Error(message);
  }
  if (!payload || payload.code !== 0) {
    throw new Error(payload?.message || '回路识别失败');
  }
  return payload.data || {};
};

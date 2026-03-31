import { apiFetch, apiPostJson } from './client.js';

export const fetchPidChartData = async (payload) => {
  return apiPostJson('/api/tuning/pid-chart-data', payload ?? {});
};

export const fetchPidPredictionCurve = async (payload) => {
  return apiPostJson('/api/tuning/pid-prediction-curve', payload ?? {});
};

export const startTuneStream = async (formData) => {
  const res = await apiFetch('/api/tune_stream', {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
  return res;
};


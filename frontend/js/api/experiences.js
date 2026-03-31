import { apiFetch, apiGetJson } from './client.js';

export const clearExperiences = async () => {
  const res = await apiFetch('/api/experiences/actions/clear', { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json().catch(() => ({}));
};

export const fetchExperienceStats = async () => {
  return apiGetJson('/api/experiences/stats');
};

export const fetchExperiences = async (params) => {
  const qs = params ? `?${params.toString()}` : '';
  return apiGetJson(`/api/experiences${qs}`);
};

export const fetchExperienceDetail = async (experienceId) => {
  return apiGetJson(`/api/experiences/${encodeURIComponent(experienceId)}`);
};

export const searchExperiences = async (formData) => {
  const res = await apiFetch('/api/experiences/search', {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};


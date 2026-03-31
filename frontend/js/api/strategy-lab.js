import { apiFetch, apiGetJson, apiPostJson } from './client.js';

export const fetchStrategyLabCases = async () => {
  return apiGetJson('/api/strategy-lab/cases');
};

export const fetchStrategyLabCandidates = async () => {
  return apiGetJson('/api/strategy-lab/candidates');
};

export const fetchStrategyLabCandidateDetail = async (candidateId) => {
  return apiGetJson(`/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}`);
};

export const generateStrategyLabCandidate = async (payload) => {
  return apiPostJson('/api/strategy-lab/candidates/generate', payload ?? {});
};

export const evaluateStrategyLabCandidate = async (candidateId) => {
  const res = await apiFetch(`/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}/evaluate`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json().catch(() => ({}));
};

export const cloneStrategyLabCandidate = async (candidateId) => {
  const res = await apiFetch(`/api/strategy-lab/candidates/${encodeURIComponent(candidateId)}/clone`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};


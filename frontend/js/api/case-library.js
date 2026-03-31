import { apiGetJson } from './client.js';

export const fetchCaseLibraryStats = async () => {
  return apiGetJson('/api/case-library/stats');
};

export const fetchCaseLibraryItems = async (params) => {
  const qs = params ? `?${params.toString()}` : '';
  return apiGetJson(`/api/case-library${qs}`);
};

export const fetchCaseLibraryDetail = async (caseId) => {
  return apiGetJson(`/api/case-library/${encodeURIComponent(caseId)}`);
};


import { apiFetch, apiGetJson } from '../api/client.js';

const MAX_LOCAL_SESSIONS = 8;
const MAX_LOCAL_MESSAGES_PER_SESSION = 12;

const isQuotaExceededError = (error) => {
  if (!error) return false;
  const name = String(error?.name || '');
  const message = String(error?.message || '');
  return name === 'QuotaExceededError'
    || message.includes('exceeded the quota')
    || message.includes('QuotaExceededError');
};

const buildResultSummary = (result) => {
  if (!result || typeof result !== 'object') return null;
  const model = result.model || {};
  const pid = result.pidParams || {};
  const evaluation = result.evaluation || {};
  return {
    __summary__: true,
    modelType: model.modelType || '',
    strategy: pid.strategyUsed || pid.strategy || '',
    finalRating: evaluation.final_rating ?? null,
    passed: evaluation.passed ?? null
  };
};

const slimTool = (tool) => {
  if (!tool || typeof tool !== 'object') return tool;
  return {
    ...tool,
    result: tool.result ? { __trimmed__: true } : tool.result
  };
};

const slimMessage = (message) => {
  if (!message || typeof message !== 'object') return message;

  if (message.type === 'result') {
    return {
      ...message,
      data: buildResultSummary(message.data)
    };
  }

  if (message.type === 'agent_turn') {
    return {
      ...message,
      tools: Array.isArray(message.tools) ? message.tools.map(slimTool) : []
    };
  }

  return message;
};

const slimSession = (session, index) => {
  if (!session || typeof session !== 'object') return session;
  const messages = Array.isArray(session.messages) ? session.messages : [];
  const keepLatestResult = index < 2;
  return {
    ...session,
    messages: messages.slice(-MAX_LOCAL_MESSAGES_PER_SESSION).map(slimMessage),
    latestResult: keepLatestResult ? session.latestResult : buildResultSummary(session.latestResult)
  };
};

const buildLocalStoragePayload = (payload, level = 0) => {
  if (!payload || typeof payload !== 'object') return payload;
  const items = Array.isArray(payload.items) ? payload.items : [];

  if (level <= 0) {
    return payload;
  }

  if (level === 1) {
    return {
      ...payload,
      items: items.slice(0, MAX_LOCAL_SESSIONS).map((session, index) => slimSession(session, index))
    };
  }

  return {
    ...payload,
    items: items.slice(0, 4).map((session, index) => ({
      id: session?.id || '',
      title: session?.title || '',
      createdAt: session?.createdAt || '',
      updatedAt: session?.updatedAt || '',
      status: session?.status || 'draft',
      context: session?.context || {},
      messages: [],
      messageIdCounter: Number(session?.messageIdCounter || 0),
      latestResult: index === 0 ? buildResultSummary(session?.latestResult) : null
    }))
  };
};

const persistLocalTaskSessions = (storageKey, payload) => {
  let lastError = null;
  for (const level of [0, 1, 2]) {
    try {
      const localPayload = buildLocalStoragePayload(payload, level);
      localStorage.setItem(storageKey, JSON.stringify(localPayload));
      if (level > 0) {
        console.warn('[task-sessions] localStorage payload trimmed', { level });
      }
      return;
    } catch (error) {
      lastError = error;
      if (!isQuotaExceededError(error)) {
        throw error;
      }
    }
  }

  if (lastError) {
    console.warn('[task-sessions] localStorage unavailable after trimming', {
      errorName: lastError?.name || '',
      errorMessage: String(lastError?.message || '')
    });
  }
};

export const loadTaskSessionsPayload = async (storageKey) => {
  let payload = null;

  try {
    const remote = await apiGetJson('/api/task-sessions');
    if (remote && Array.isArray(remote.items)) {
      payload = remote;
      try {
        persistLocalTaskSessions(storageKey, remote);
      } catch (_) {}
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
  try {
    persistLocalTaskSessions(storageKey, payload);
  } catch (error) {
    console.warn('[task-sessions] failed to persist local copy', {
      errorName: error?.name || '',
      errorMessage: String(error?.message || '')
    });
  }
  try {
    await apiFetch('/api/task-sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (_) {}
};

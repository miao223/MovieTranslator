async function request(url, options = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      detail = body.detail || detail
    } catch { /* keep statusText */ }
    throw new Error(detail)
  }
  return resp.json()
}

export const api = {
  createJob: (payload) =>
    request('/api/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  getJob: (id) => request(`/api/jobs/${id}`),
  batchScan: (path, recursive, skipExisting) =>
    request(`/api/batch/scan?path=${encodeURIComponent(path)}&recursive=${recursive}&skip_existing=${skipExisting}`),
  createBatch: (payload) =>
    request('/api/batch', { method: 'POST', body: JSON.stringify(payload) }),
  getBatch: (id) => request(`/api/batch/${id}`),
  cancelBatch: (id) => request(`/api/batch/${id}/cancel`, { method: 'POST' }),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  getSettings: () => request('/api/settings'),
  saveSettings: (settings) =>
    request('/api/settings', { method: 'PUT', body: JSON.stringify(settings) }),
  testLLM: (llm) =>
    request('/api/settings/test-llm', { method: 'POST', body: JSON.stringify(llm) }),
  browse: (path) =>
    request(`/api/fs/browse?path=${encodeURIComponent(path || '')}`),
  resolvePath: (path) =>
    request(`/api/fs/resolve?path=${encodeURIComponent(path)}`),
  quickAccess: () => request('/api/fs/quick-access'),
  promptPreview: (payload) =>
    request('/api/prompts/preview', { method: 'POST', body: JSON.stringify(payload) }),
  modelStatus: (modelSize) =>
    request(`/api/asr/model-status?model_size=${encodeURIComponent(modelSize)}`),
  cudaStatus: () => request('/api/asr/cuda-status'),
  storageInfo: () => request('/api/asr/storage-info'),
  downloadModel: (modelSize) =>
    request('/api/asr/download', { method: 'POST', body: JSON.stringify({ model_size: modelSize }) }),
  downloadStatus: (modelSize) =>
    request(`/api/asr/download-status?model_size=${encodeURIComponent(modelSize)}`),
  resultUrl: (id) => `/api/jobs/${id}/result`,
  eventsUrl: (id) => `/api/jobs/${id}/events`,
}

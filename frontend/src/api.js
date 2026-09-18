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
  getVersion: () => request('/api/version'),
  createJob: (payload) =>
    request('/api/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  getJob: (id) => request(`/api/jobs/${id}`),
  // targetLanguage decides what counts as "already translated": a subtitle
  // in that language means done, one in any other language is material
  batchScan: (path, recursive, skipExisting, targetLanguage = '') =>
    request(`/api/batch/scan?path=${encodeURIComponent(path)}&recursive=${recursive}`
            + `&skip_existing=${skipExisting}`
            + `&target_language=${encodeURIComponent(targetLanguage)}`),
  createBatch: (payload) =>
    request('/api/batch', { method: 'POST', body: JSON.stringify(payload) }),
  getBatch: (id) => request(`/api/batch/${id}`),
  cancelBatch: (id) => request(`/api/batch/${id}/cancel`, { method: 'POST' }),
  saveBatchGlossary: (id) =>
    request(`/api/batch/${id}/glossary/save`, { method: 'POST' }),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  getSettings: () => request('/api/settings'),
  saveSettings: (settings) =>
    request('/api/settings', { method: 'PUT', body: JSON.stringify(settings) }),
  testLLM: (llm) =>
    request('/api/settings/test-llm', { method: 'POST', body: JSON.stringify(llm) }),
  testVision: (llm) =>
    request('/api/settings/test-vision', { method: 'POST', body: JSON.stringify(llm) }),
  testAsrApi: (llm) =>
    request('/api/settings/test-asr-api', { method: 'POST', body: JSON.stringify(llm) }),
  browse: (path) =>
    request(`/api/fs/browse?path=${encodeURIComponent(path || '')}`),
  resolvePath: (path) =>
    request(`/api/fs/resolve?path=${encodeURIComponent(path)}`),
  quickAccess: () => request('/api/fs/quick-access'),
  audioTracks: (path) =>
    request(`/api/media/audio-tracks?path=${encodeURIComponent(path)}`),
  subtitleTracks: (path) =>
    request(`/api/media/subtitle-tracks?path=${encodeURIComponent(path)}`),
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
  encoders: () => request('/api/media/encoders'),
  serverInfo: () => request('/api/server/info'),
  storageUsage: () => request('/api/storage/usage'),
  clearCheckpoints: () =>
    request('/api/storage/clear-checkpoints', { method: 'POST' }),
  regenerateToken: () =>
    request('/api/server/token/regenerate', { method: 'POST' }),
  // part='original' 取双语分离模式的原文那一份；不传时链接与从前逐字节相同
  resultUrl: (id, part = '') =>
    `/api/jobs/${id}/result${part ? `?part=${part}` : ''}`,
  jobLogUrl: (id) => `/api/logs/job/${id}`,
  logs: () => request('/api/logs'),
  eventsUrl: (id) => `/api/jobs/${id}/events`,

  // ------------------------------------------------------------- 列队
  queue: () => request('/api/queue'),
  enqueueJob: (payload) =>
    request('/api/queue/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueBatch: (payload) =>
    request('/api/queue/batch', { method: 'POST', body: JSON.stringify(payload) }),
  pauseQueue: (paused) =>
    request('/api/queue/pause', { method: 'POST', body: JSON.stringify({ paused }) }),
  reorderQueue: (ids) =>
    request('/api/queue/order', { method: 'PUT', body: JSON.stringify({ ids }) }),
  clearFinished: () => request('/api/queue/finished', { method: 'DELETE' }),
  queueEntrySettings: (id) => request(`/api/queue/${id}/settings`),
  cancelQueueEntry: (id) => request(`/api/queue/${id}/cancel`, { method: 'POST' }),
  retryQueueEntry: (id, fresh = false) =>
    request(`/api/queue/${id}/retry`, { method: 'POST', body: JSON.stringify({ fresh }) }),
  removeQueueEntry: (id) => request(`/api/queue/${id}`, { method: 'DELETE' }),
  // an <a download> href, so it cannot be a fetch: /api/jobs/{id}/result
  // 404s once a restart has emptied the in-memory job table
  queueResultUrl: (id, part = '') =>
    `/api/queue/${id}/result${part ? `?part=${part}` : ''}`,
}

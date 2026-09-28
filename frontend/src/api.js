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
  batchScan: (path, recursive, skipExisting, targetLanguage = '', subtitleLanguage = '') =>
    request(`/api/batch/scan?path=${encodeURIComponent(path)}&recursive=${recursive}`
            + `&skip_existing=${skipExisting}`
            + `&target_language=${encodeURIComponent(targetLanguage)}`
            // 同一部片旁边有好几种语言的字幕时，这决定挑哪一份当原文
            + `&subtitle_language=${encodeURIComponent(subtitleLanguage)}`),
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
  testAsrApi: (llm, flex = false) =>
    request(`/api/settings/test-asr-api?flex=${flex ? 'true' : 'false'}`,
            { method: 'POST', body: JSON.stringify(llm) }),
  // mode='disc' lists .iso images and names which folders are discs
  browse: (path, mode = '') =>
    request(`/api/fs/browse?path=${encodeURIComponent(path || '')}`
            + (mode ? `&mode=${mode}` : '')),
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

  // ------------------------------------------------------------- 原盘
  // Asked again whenever a switch on the page changes: it reads only the
  // disc's metadata, so it answers in milliseconds.
  // null = let the server decide: 整片/分集 from the disc, the episode
  // number from the volumes before it, the output place from the settings
  discScan: ({ path, series = null, episodeStart = null, name = '', outputMode = null,
               outputDir = '' }) =>
    request(`/api/disc/scan?path=${encodeURIComponent(path)}`
            + (series === null ? '' : `&series=${series}`)
            + (episodeStart === null ? '' : `&episode_start=${episodeStart}`)
            + `&name=${encodeURIComponent(name)}`
            + (outputMode === null ? '' : `&output_mode=${outputMode}`)
            + `&output_dir=${encodeURIComponent(outputDir)}`),
  enqueueDisc: (payload) =>
    request('/api/queue/disc', { method: 'POST', body: JSON.stringify(payload) }),
  // the batch mode: every disc in a folder; asked again with the answers
  // given so far whenever one changes
  discBatchScan: (payload) =>
    request('/api/disc/batch-scan', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueDiscBatch: (payload) =>
    request('/api/queue/disc-batch', { method: 'POST', body: JSON.stringify(payload) }),

  // ------------------------------------------------------------- 压制
  // one file: its streams, plus a few decoded frames' worth of detail
  encodeProbe: (path) => request(`/api/encode/probe?path=${encodeURIComponent(path)}`),
  // a folder: every video in it (headers only), never the insides of a disc
  encodeScan: (payload) =>
    request('/api/encode/scan', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueEncode: (payload) =>
    request('/api/queue/encode', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueEncodeBatch: (payload) =>
    request('/api/queue/encode-batch', { method: 'POST', body: JSON.stringify(payload) }),
  audioProbe: (path) => request(`/api/audio/probe?path=${encodeURIComponent(path)}`),
  audioScan: (payload) =>
    request('/api/audio/scan', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueAudio: (payload) =>
    request('/api/queue/audio', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueAudioBatch: (payload) =>
    request('/api/queue/audio-batch', { method: 'POST', body: JSON.stringify(payload) }),

  // ------------------------------------------------------------- 列队
  queue: () => request('/api/queue'),
  enqueueJob: (payload) =>
    request('/api/queue/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  enqueueBatch: (payload) =>
    request('/api/queue/batch', { method: 'POST', body: JSON.stringify(payload) }),
  pauseQueue: (paused) =>
    request('/api/queue/pause', { method: 'POST', body: JSON.stringify({ paused }) }),
  cpuYield: (enabled) =>
    request('/api/queue/cpu-yield', { method: 'POST', body: JSON.stringify({ enabled }) }),
  encodePick: (body) =>
    request('/api/encode/pick', { method: 'POST', body: JSON.stringify(body) }),
  memoryLimit: (gb) =>
    request('/api/queue/memory-limit', { method: 'POST', body: JSON.stringify({ gb }) }),
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

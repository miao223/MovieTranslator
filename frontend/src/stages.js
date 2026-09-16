// Stage names and their colours, shared by every view that shows a job.
//
// A plain module of constants, like theme.js: there is no components/ dir
// and no store, and three views now render the same stage in the same tag.

export const STAGE_LABELS = {
  queued: '等待中',
  pending: '排队中',
  extracting: '提取音频',
  importing: '读取字幕',
  transcribing: '语音识别',
  refining: '转写预处理',
  translating: 'AI 翻译',
  composing: '生成字幕',
  running: '进行中',
  done: '完成',
  failed: '失败',
  cancelled: '已取消',
}

export const TERMINAL_STAGES = ['done', 'failed', 'cancelled']

export function stageLabel(stage) {
  return STAGE_LABELS[stage] || stage
}

export function stageTagType(stage) {
  if (stage === 'done') return 'success'
  if (stage === 'failed') return 'danger'
  if (stage === 'cancelled' || stage === 'pending' || stage === 'queued') return 'info'
  return 'primary'
}

export function isTerminal(stage) {
  return TERMINAL_STAGES.includes(stage)
}

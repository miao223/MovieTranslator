<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'
import { stageLabel, stageTagType } from '../stages'

const props = defineProps({ active: { type: Boolean, default: false } })
const emit = defineEmits(['changed', 'goto'])

const queue = ref(null)
const logs = ref([])
const logBox = ref(null)
const filter = ref('all')
const busy = ref(false)
const settingsDialog = ref({ visible: false, entry: null, body: null })

let timer = null
let eventSource = null
let attachedJob = ''

// -------------------------------------------------------------- polling
//
// This view is meant to sit in a background tab all night, so the timer is
// gated on being visible AND on the tab being the active one — `lazy` on
// the pane only defers the first mount, it does not stop the interval once
// the pane has been visited.

function interval() {
  return queue.value?.entries?.some((e) => e.status === 'running') ? 2000 : 10000
}

async function refresh() {
  try {
    queue.value = await api.queue()
    attachLog()
  } catch {
    /* transient poll error */
  }
}

function restartTimer() {
  clearInterval(timer)
  timer = null
  if (!props.active || document.hidden) return
  timer = setInterval(refresh, interval())
}

function onVisibility() {
  if (document.hidden) {
    closeLog()
    clearInterval(timer)
    timer = null
  } else if (props.active) {
    refresh()
    restartTimer()
  }
}

watch(() => props.active, (on) => {
  if (on) refresh()
  else closeLog()
  restartTimer()
})

onMounted(() => {
  document.addEventListener('visibilitychange', onVisibility)
  if (props.active) refresh()
  restartTimer()
})

onBeforeUnmount(() => {
  document.removeEventListener('visibilitychange', onVisibility)
  clearInterval(timer)
  closeLog()
})

// --------------------------------------------------------------- the log
//
// Only ev.log is read; state comes from the poll, so there is one source
// of truth for "is it done". The backend replays a job's whole history to
// every new subscriber, so closing and reopening loses nothing.

function closeLog() {
  eventSource?.close()
  eventSource = null
  attachedJob = ''
}

function attachLog() {
  const entry = running.value
  if (!entry?.job_id || !entry.job_live) {
    if (!entry) closeLog()
    return
  }
  if (entry.job_id === attachedJob) return
  closeLog()
  attachedJob = entry.job_id
  logs.value = []          // the box is labelled with a row; it belongs to it
  eventSource = new EventSource(api.eventsUrl(entry.job_id))
  eventSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data)
    if (!ev.log) return
    logs.value.push(ev.log)
    if (logs.value.length > 500) logs.value.splice(0, logs.value.length - 500)
    requestAnimationFrame(() => {
      if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
    })
  }
}

// ------------------------------------------------------------- derived

const entries = computed(() => queue.value?.entries || [])
const running = computed(() => entries.value.find((e) => e.status === 'running') || null)
const counts = computed(() => {
  const out = { queued: 0, running: 0, done: 0, failed: 0, cancelled: 0 }
  for (const e of entries.value) if (e.status in out) out[e.status] += 1
  return out
})

// Settings generations, lettered by first appearance in queue order — not
// relative to "current", which would re-letter every row the moment the
// settings change. When every entry shares one hash and it is the current
// one, the column is not rendered at all: the common case stays clean and
// the odd row announces itself the moment a second generation exists.
const generations = computed(() => {
  const seen = new Map()
  for (const e of entries.value) {
    if (e.settings_hash && !seen.has(e.settings_hash)) {
      seen.set(e.settings_hash, String.fromCharCode(65 + seen.size))
    }
  }
  return seen
})
const showGenerations = computed(() =>
  generations.value.size > 1 || entries.value.some((e) => e.settings_differs))

const visible = computed(() => {
  if (filter.value === 'all') return entries.value
  if (filter.value === 'waiting') return entries.value.filter((e) => e.status === 'queued')
  if (filter.value === 'failed') return entries.value.filter((e) => e.status === 'failed')
  return entries.value.filter((e) => ['done', 'cancelled'].includes(e.status))
})

const baseName = (p) => (p || '').split(/[\\/]/).pop() || p
const waitingIds = computed(() =>
  entries.value.filter((e) => e.status === 'queued').map((e) => e.id))

// ------------------------------------------------------------- commands

async function act(fn, okMessage) {
  busy.value = true
  try {
    await fn()
    if (okMessage) ElMessage.success(okMessage)
    await refresh()
    emit('changed')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    busy.value = false
  }
}

const setPaused = (paused) => act(() => api.pauseQueue(paused))
const cancelEntry = (entry) => act(() => api.cancelQueueEntry(entry.id))
const removeEntry = (entry) => act(() => api.removeQueueEntry(entry.id))
const retryEntry = (entry, fresh) =>
  act(() => api.retryQueueEntry(entry.id, fresh),
      fresh ? '已用当前设置重新加入列队' : '已用原来的设置重新加入列队')

function move(entry, delta) {
  const ids = [...waitingIds.value]
  const at = ids.indexOf(entry.id)
  const to = at + delta
  if (at < 0 || to < 0 || to >= ids.length) return
  ids.splice(to, 0, ...ids.splice(at, 1))
  act(() => api.reorderQueue(ids))
}

function toTop(entry) {
  const ids = [...waitingIds.value]
  const at = ids.indexOf(entry.id)
  if (at <= 0) return
  ids.unshift(...ids.splice(at, 1))
  act(() => api.reorderQueue(ids))
}

async function clearFinished() {
  await ElMessageBox.confirm('清除所有已完成、失败和已取消的记录？', '清除记录',
    { type: 'warning' })
  act(() => api.clearFinished(), '已清除')
}

async function showSettings(entry) {
  try {
    const body = await api.queueEntrySettings(entry.id)
    settingsDialog.value = { visible: true, entry, body }
  } catch (e) {
    ElMessage.error(e.message)
  }
}
</script>

<template>
  <div v-if="queue">
    <!-- 控制条 -->
    <el-card shadow="never" class="section">
      <template #header>🗂️ 列队</template>
      <div class="bar">
        <el-tag :type="queue.paused ? 'warning' : counts.running ? 'primary' : 'info'">
          {{ queue.paused ? '已暂停' : counts.running ? '运行中' : '空闲' }}
        </el-tag>
        <el-switch
          :model-value="queue.paused"
          active-text="暂停列队"
          :disabled="busy"
          @update:model-value="setPaused"
        />
        <span class="counts">
          等待 {{ counts.queued }} · 完成 {{ counts.done }}
          <template v-if="counts.failed"> · 失败 {{ counts.failed }}</template>
          <template v-if="counts.cancelled"> · 已取消 {{ counts.cancelled }}</template>
        </span>
        <span style="flex: 1" />
        <el-button size="small" :disabled="busy" @click="refresh">刷新</el-button>
        <el-button size="small" :disabled="busy" @click="clearFinished">清除已完成</el-button>
      </div>
      <div class="hint" style="display: block; margin-top: 6px">
        暂停后不再开始新的任务，<strong>正在跑的那一条不会被打断</strong>；
        想立刻停手就先暂停，再取消正在跑的那条。<br />
        每条任务记住的是<strong>加入列队时已保存的设置</strong>，之后改设置只影响后面加入的任务。
      </div>
    </el-card>

    <!-- 正在进行 -->
    <el-card v-if="running" shadow="never" class="section">
      <template #header>▶️ 正在进行</template>
      <div class="bar">
        <el-tag :type="stageTagType(running.stage)">{{ stageLabel(running.stage) }}</el-tag>
        <strong :title="running.title">{{ baseName(running.title) }}</strong>
        <el-progress :percentage="Math.round(running.progress)" style="flex: 1; min-width: 160px" />
        <el-button size="small" :disabled="busy" @click="cancelEntry(running)">取消这条</el-button>
      </div>
      <div v-if="running.stage === 'pending'" class="hint" style="display: block">
        正在等另一个任务跑完（同时只跑一个）。
      </div>
      <div v-else-if="running.message" class="hint" style="display: block">{{ running.message }}</div>
      <div v-if="logs.length" ref="logBox" class="logs">
        <div v-for="(line, i) in logs" :key="i">{{ line }}</div>
      </div>
    </el-card>

    <!-- 列表 -->
    <el-card shadow="never" class="section">
      <template #header>
        <div class="bar">
          <span>列队内容</span>
          <span style="flex: 1" />
          <el-radio-group v-model="filter" size="small">
            <el-radio-button value="all">全部</el-radio-button>
            <el-radio-button value="waiting">等待中</el-radio-button>
            <el-radio-button value="finished">已完成</el-radio-button>
            <el-radio-button value="failed">失败</el-radio-button>
          </el-radio-group>
        </div>
      </template>

      <el-empty v-if="!entries.length" description="列队是空的">
        <div class="hint" style="display: block; max-width: 460px; margin-bottom: 12px">
          在「翻译任务」页填好表单后点「加入列队」。每条记录都会记住加入时
          <strong>已保存</strong>的设置，之后改设置不影响它。
        </div>
        <el-button type="primary" @click="emit('goto', 'home')">去添加</el-button>
      </el-empty>
      <el-empty v-else-if="!visible.length" description="这个筛选下没有记录" />

      <div v-else class="queue-list">
        <div v-for="(entry, i) in visible" :key="entry.id" class="queue-row">
          <span class="idx">#{{ i + 1 }}</span>
          <el-tag size="small" :type="stageTagType(entry.stage || entry.status)">
            {{ stageLabel(entry.stage || entry.status) }}
          </el-tag>
          <span class="title" :title="entry.title">{{ baseName(entry.title) }}</span>
          <span class="summary">{{ entry.summary }}</span>
          <el-tag
            v-if="showGenerations && entry.settings_hash"
            size="small"
            :type="entry.settings_differs ? 'warning' : 'info'"
            class="gen"
            @click="showSettings(entry)"
          >
            设置 {{ generations.get(entry.settings_hash) }}
          </el-tag>
          <el-tag
            v-if="entry.interrupted"
            size="small"
            type="warning"
            :title="`这条任务被中断过 ${entry.interrupted} 次，每次都是从头重跑的`"
          >
            中断 ×{{ entry.interrupted }}
          </el-tag>
          <el-progress
            v-if="entry.status === 'running'"
            :percentage="Math.round(entry.progress)"
            style="width: 120px"
          />
          <span class="actions">
            <template v-if="entry.status === 'queued'">
              <el-button size="small" text :disabled="busy" @click="move(entry, -1)">↑</el-button>
              <el-button size="small" text :disabled="busy" @click="move(entry, 1)">↓</el-button>
              <el-button size="small" text :disabled="busy" @click="toTop(entry)">置顶</el-button>
              <el-button size="small" text :disabled="busy" @click="cancelEntry(entry)">取消</el-button>
            </template>
            <el-button
              v-else-if="entry.status === 'running'"
              size="small" text :disabled="busy" @click="cancelEntry(entry)"
            >取消</el-button>
            <template v-else>
              <el-button size="small" text :disabled="busy" @click="retryEntry(entry, false)">重试</el-button>
              <el-button
                v-if="entry.result_srt || entry.result_video"
                size="small" text tag="a" :href="api.queueResultUrl(entry.id)" download
              >下载</el-button>
              <el-button
                v-if="entry.has_log"
                size="small" text tag="a" :href="api.jobLogUrl(entry.job_id)" download
              >日志</el-button>
              <el-button size="small" text :disabled="busy" @click="removeEntry(entry)">移除</el-button>
            </template>
          </span>
          <div v-if="entry.error" class="row-note error">{{ entry.error }}</div>
          <div v-else-if="entry.note" class="row-note">{{ entry.note }}</div>
        </div>
      </div>
    </el-card>

    <!-- 某一条冻结的设置 -->
    <el-dialog v-model="settingsDialog.visible" title="这条任务冻结的设置" width="640px">
      <div v-if="settingsDialog.body" class="hint" style="display: block; margin-bottom: 8px">
        <template v-if="settingsDialog.body.unreadable">
          这条记录的设置快照读不出来，运行时会使用当前设置。
        </template>
        <template v-else-if="settingsDialog.body.same_as_current">
          与当前设置相同。
        </template>
        <template v-else>
          与当前设置有 {{ settingsDialog.body.differs.length }} 处不同（下面标出来的行）。
        </template>
      </div>
      <div class="logs snapshot">
        <div
          v-for="(line, i) in settingsDialog.body?.lines || []"
          :key="i"
          :class="{ differs: settingsDialog.body?.differs?.includes(line) }"
        >{{ line }}</div>
      </div>
      <template #footer>
        <el-button @click="settingsDialog.visible = false">关闭</el-button>
        <el-button
          v-if="settingsDialog.entry && ['done', 'failed', 'cancelled'].includes(settingsDialog.entry.status)"
          @click="retryEntry(settingsDialog.entry, false); settingsDialog.visible = false"
        >用这份设置重试</el-button>
        <el-button
          v-if="settingsDialog.entry && ['done', 'failed', 'cancelled'].includes(settingsDialog.entry.status)"
          type="primary"
          @click="retryEntry(settingsDialog.entry, true); settingsDialog.visible = false"
        >用当前设置重试</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.section {
  margin-bottom: 16px;
}
.bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.counts {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.hint {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.queue-list {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--app-radius);
  max-height: 60vh;
  overflow-y: auto;
}
.queue-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  font-size: 13px;
  flex-wrap: wrap;
}
.queue-row:last-child {
  border-bottom: none;
}
.idx {
  color: var(--el-text-color-secondary);
  font-family: var(--app-mono);
  min-width: 34px;
}
.title {
  flex: 1;
  min-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.summary {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.gen {
  cursor: pointer;
}
.actions {
  display: flex;
  gap: 2px;
}
.row-note {
  flex-basis: 100%;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  padding-left: 44px;
}
.row-note.error {
  color: var(--el-color-danger);
}
.logs {
  height: 200px;
  overflow-y: auto;
  padding: 8px 10px;
  margin-top: 10px;
  background: var(--app-log-bg);
  color: var(--app-log-fg);
  font-family: var(--app-mono);
  font-size: 12px;
  border-radius: var(--app-radius);
}
.logs.snapshot {
  height: 380px;
  margin-top: 0;
}
.logs .differs {
  color: var(--el-color-warning);
}
</style>

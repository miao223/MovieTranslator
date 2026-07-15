<script setup>
import { onBeforeUnmount, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const SOURCE_LANGS = [
  { value: 'auto', label: '自动检测' },
  { value: 'en', label: '英语' },
  { value: 'ja', label: '日语' },
  { value: 'ko', label: '韩语' },
  { value: 'fr', label: '法语' },
  { value: 'de', label: '德语' },
  { value: 'es', label: '西班牙语' },
  { value: 'ru', label: '俄语' },
  { value: 'zh', label: '中文' },
]

const TARGET_LANGS = ['简体中文', '繁體中文', 'English', '日本語', '한국어', 'Français', 'Deutsch']

const mode = ref('single') // 'single' | 'batch'

const form = reactive({
  video_path: '',
  source_language: 'auto',
  target_language: '简体中文',
  synopsis: '',
  output_mode: 'bilingual',
})

const batchForm = reactive({
  directory: '',
  recursive: true,
  skip_existing_srt: true,
})

// ---------------------------------------------------------- file browser
const browserVisible = ref(false)
const browser = reactive({ path: '', parent: null, dirs: [], files: [] })
const addressInput = ref('')
const quickAccess = ref([])
const browsePick = ref('file') // 'file' | 'dir' — what the dialog selects

function pickThisDir() {
  if (!browser.path) return
  batchForm.directory = browser.path
  browserVisible.value = false
}

async function openBrowser(path = '', pick = null) {
  if (pick) browsePick.value = pick
  try {
    const data = await api.browse(path)
    Object.assign(browser, data)
    addressInput.value = data.path
    browserVisible.value = true
    if (!quickAccess.value.length) {
      api.quickAccess().then((r) => (quickAccess.value = r.items)).catch(() => {})
    }
  } catch (e) {
    ElMessage.error(e.message)
  }
}

// jump to a pasted Explorer path: a folder opens it, a full video file
// path selects it directly (quotes from "复制文件地址" are stripped server-side)
async function jumpToAddress() {
  const raw = addressInput.value.trim()
  if (!raw) return
  try {
    const r = await api.resolvePath(raw)
    if (r.type === 'dir') {
      openBrowser(r.path)
    } else if (r.type === 'file' && r.is_video && browsePick.value === 'file') {
      form.video_path = r.path
      browserVisible.value = false
      ElMessage.success('已选择: ' + r.path)
    } else if (r.type === 'file' && browsePick.value === 'dir') {
      ElMessage.warning('当前在选择目录，请粘贴文件夹路径或点「选择此目录」')
    } else if (r.type === 'file') {
      ElMessage.warning('该文件不是支持的视频格式')
    } else {
      ElMessage.warning('路径不存在: ' + r.path)
    }
  } catch (e) {
    ElMessage.error(e.message)
  }
}

function joinPath(dir, name) {
  if (!dir) return name
  const sep = dir.includes('\\') ? '\\' : '/'
  return dir.endsWith(sep) ? dir + name : dir + sep + name
}

function pickFile(name) {
  if (browsePick.value === 'dir') return
  form.video_path = joinPath(browser.path, name)
  browserVisible.value = false
}

function fmtSize(bytes) {
  if (bytes > 1 << 30) return (bytes / (1 << 30)).toFixed(1) + ' GB'
  if (bytes > 1 << 20) return (bytes / (1 << 20)).toFixed(1) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

// ---------------------------------------------------------------- job
const job = ref(null) // { id, stage, progress, message }
const logs = ref([])
const logBox = ref(null)
let eventSource = null

const STAGE_LABELS = {
  pending: '排队中',
  extracting: '提取音频',
  transcribing: '语音识别',
  translating: 'AI 翻译',
  composing: '生成字幕',
  done: '完成',
  failed: '失败',
  cancelled: '已取消',
}

const running = () =>
  job.value && !['done', 'failed', 'cancelled'].includes(job.value.stage)

async function start() {
  if (!form.video_path) {
    ElMessage.warning('请先选择视频文件')
    return
  }
  try {
    const status = await api.createJob({ ...form })
    job.value = { ...status }
    logs.value = []
    listen(status.id)
  } catch (e) {
    ElMessage.error(e.message)
  }
}

function listen(id) {
  eventSource?.close()
  eventSource = new EventSource(api.eventsUrl(id))
  eventSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data)
    job.value = { ...job.value, ...ev }
    if (ev.log) {
      const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
      logs.value.push(`[${ts}] ${ev.log}`)
      if (logs.value.length > 500) logs.value.splice(0, logs.value.length - 500)
      requestAnimationFrame(() => {
        if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
      })
    }
    if (['done', 'failed', 'cancelled'].includes(ev.stage)) {
      eventSource.close()
      if (ev.stage === 'done') {
        ElMessage.success('字幕生成完成！')
        // SSE events carry no srt fields; fetch the final status once
        api.getJob(job.value.id).then((s) => (job.value = { ...job.value, ...s })).catch(() => {})
      }
      if (ev.stage === 'failed') ElMessage.error(ev.message)
    }
  }
  eventSource.onerror = () => {
    if (running()) ElMessage.warning('进度连接中断，正在重连…')
  }
}

async function cancel() {
  if (job.value) await api.cancelJob(job.value.id)
}

// ---------------------------------------------------------------- batch
const batch = ref(null)
const scanResult = ref(null)
const confirmVisible = ref(false)
let batchTimer = null
let currentSseJob = ''

const batchRunning = () =>
  batch.value && batch.value.pending + batch.value.running > 0

async function startBatchScan() {
  if (!batchForm.directory) {
    ElMessage.warning('请先选择目录')
    return
  }
  try {
    const r = await api.batchScan(
      batchForm.directory, batchForm.recursive, batchForm.skip_existing_srt,
    )
    if (!r.total) {
      ElMessage.warning(
        '目录中没有需要翻译的视频'
        + (r.skipped.length ? `（${r.skipped.length} 个已有字幕，被跳过）` : ''),
      )
      return
    }
    scanResult.value = r
    confirmVisible.value = true
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function startBatch() {
  confirmVisible.value = false
  try {
    batch.value = await api.createBatch({
      ...batchForm,
      source_language: form.source_language,
      target_language: form.target_language,
      synopsis: form.synopsis,
      output_mode: form.output_mode,
    })
    job.value = null
    logs.value = []
    currentSseJob = ''
    pollBatch()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

function pollBatch() {
  clearInterval(batchTimer)
  batchTimer = setInterval(async () => {
    try {
      const b = await api.getBatch(batch.value.id)
      batch.value = b
      // re-attach the log stream whenever the running job changes
      if (b.current_job_id && b.current_job_id !== currentSseJob) {
        currentSseJob = b.current_job_id
        attachBatchLog(b.current_job_id)
      }
      if (b.pending + b.running === 0) {
        clearInterval(batchTimer)
        eventSource?.close()
        ElMessage.success(`批量完成：成功 ${b.done}，失败 ${b.failed}，取消 ${b.cancelled}`)
      }
    } catch { /* transient poll error */ }
  }, 2000)
}

function attachBatchLog(id) {
  eventSource?.close()
  eventSource = new EventSource(api.eventsUrl(id))
  eventSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data)
    if (ev.log) {
      const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
      logs.value.push(`[${ts}] ${ev.log}`)
      if (logs.value.length > 500) logs.value.splice(0, logs.value.length - 500)
      requestAnimationFrame(() => {
        if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
      })
    }
  }
}

async function cancelBatch() {
  if (batch.value) await api.cancelBatch(batch.value.id)
}

function baseName(p) {
  return p.split(/[\\/]/).pop()
}

onBeforeUnmount(() => {
  eventSource?.close()
  clearInterval(batchTimer)
})
</script>

<template>
  <el-card shadow="never">
    <el-form :model="form" label-width="110px">
      <el-form-item label="翻译对象">
        <el-radio-group v-model="mode">
          <el-radio value="single">单个文件</el-radio>
          <el-radio value="batch">批量目录</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item v-if="mode === 'single'" label="视频文件" required>
        <el-input v-model="form.video_path" placeholder="视频文件的完整路径">
          <template #append>
            <el-button @click="openBrowser('', 'file')">浏览…</el-button>
          </template>
        </el-input>
      </el-form-item>
      <template v-else>
        <el-form-item label="视频目录" required>
          <el-input v-model="batchForm.directory" placeholder="将翻译该目录内的所有视频文件">
            <template #append>
              <el-button @click="openBrowser('', 'dir')">浏览…</el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item label="批量选项">
          <el-switch v-model="batchForm.recursive" />
          <span class="hint" style="margin-right: 24px">包含子目录</span>
          <el-switch v-model="batchForm.skip_existing_srt" />
          <span class="hint">跳过已有同名字幕的视频</span>
        </el-form-item>
      </template>
      <el-form-item label="音频语言">
        <el-select v-model="form.source_language" style="width: 200px">
          <el-option v-for="l in SOURCE_LANGS" :key="l.value" :value="l.value" :label="l.label" />
        </el-select>
        <span class="hint">语音识别的源语言，不确定就选自动检测</span>
      </el-form-item>
      <el-form-item label="目标语言">
        <el-select v-model="form.target_language" style="width: 200px">
          <el-option v-for="l in TARGET_LANGS" :key="l" :value="l" :label="l" />
        </el-select>
      </el-form-item>
      <el-form-item label="剧情简介">
        <el-input
          v-model="form.synopsis"
          type="textarea"
          :rows="3"
          placeholder="（可选）填写电影剧情简介、人物关系等，可显著提升译名与语气的准确性"
        />
      </el-form-item>
      <el-form-item label="字幕形式">
        <el-radio-group v-model="form.output_mode">
          <el-radio value="bilingual">双语（原文 + 译文）</el-radio>
          <el-radio value="translation_only">纯译文</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item>
        <el-button
          type="primary"
          :disabled="running() || batchRunning()"
          @click="mode === 'single' ? start() : startBatchScan()"
        >
          开始翻译
        </el-button>
        <el-button v-if="running()" type="danger" plain @click="cancel">取消任务</el-button>
        <el-button v-if="batchRunning()" type="danger" plain @click="cancelBatch">取消批量</el-button>
      </el-form-item>
    </el-form>
  </el-card>

  <el-card v-if="batch" shadow="never" class="progress-card">
    <div class="progress-row">
      <el-tag :type="batchRunning() ? 'primary' : batch.failed ? 'warning' : 'success'">
        批量 {{ batch.done + batch.failed + batch.cancelled }}/{{ batch.total }}
      </el-tag>
      <el-progress
        :percentage="Math.round(((batch.done + batch.failed + batch.cancelled) / batch.total) * 100)"
        :status="!batchRunning() ? (batch.failed ? 'warning' : 'success') : undefined"
        style="flex: 1"
      />
    </div>
    <p class="message">
      成功 {{ batch.done }} · 失败 {{ batch.failed }} · 取消 {{ batch.cancelled }}
      · 排队 {{ batch.pending }}
      <span v-if="batch.skipped.length"> · 跳过 {{ batch.skipped.length }}（已有字幕）</span>
    </p>
    <div class="batch-files">
      <div v-for="j in batch.jobs" :key="j.id" class="batch-file">
        <span class="name">{{ baseName(j.video_path) }}</span>
        <el-progress
          v-if="!['done', 'failed', 'cancelled', 'pending'].includes(j.stage)"
          :percentage="Math.round(j.progress)" style="width: 140px"
        />
        <el-tag
          size="small"
          :type="j.stage === 'done' ? 'success' : j.stage === 'failed' ? 'danger' : j.stage === 'cancelled' ? 'info' : j.stage === 'pending' ? 'info' : 'primary'"
        >
          {{ STAGE_LABELS[j.stage] || j.stage }}
        </el-tag>
      </div>
    </div>
    <div ref="logBox" class="logs">
      <div v-for="(line, i) in logs" :key="i" class="log-line">{{ line }}</div>
    </div>
  </el-card>

  <el-card v-if="job" shadow="never" class="progress-card">
    <div class="progress-row">
      <el-tag :type="job.stage === 'failed' ? 'danger' : job.stage === 'done' ? 'success' : 'primary'">
        {{ STAGE_LABELS[job.stage] || job.stage }}
      </el-tag>
      <el-progress
        :percentage="Math.round(job.progress || 0)"
        :status="job.stage === 'failed' ? 'exception' : job.stage === 'done' ? 'success' : undefined"
        style="flex: 1"
      />
    </div>
    <p class="message">{{ job.message }}</p>
    <div ref="logBox" class="logs">
      <div v-for="(line, i) in logs" :key="i" class="log-line">{{ line }}</div>
    </div>
    <template v-if="job.stage === 'done' && job.srt_filename">
      <el-alert v-if="job.srt_in_place" type="success" :closable="false" style="margin-bottom: 8px">
        字幕已保存到视频所在目录：<code>{{ job.srt_filename }}</code>
      </el-alert>
      <el-alert v-else type="warning" :closable="false" style="margin-bottom: 8px">
        视频目录不可写，字幕暂存在工作目录，请下载保存
      </el-alert>
      <el-button type="success" tag="a" :href="api.resultUrl(job.id)" download>
        下载 SRT 字幕
      </el-button>
    </template>
  </el-card>

  <el-dialog v-model="confirmVisible" title="确认批量翻译" width="560px">
    <p v-if="scanResult">
      共找到 <strong>{{ scanResult.total }}</strong> 个待翻译视频<span v-if="scanResult.skipped.length">，另有 {{ scanResult.skipped.length }} 个已有字幕将被跳过</span>：
    </p>
    <div v-if="scanResult" class="scan-list">
      <div v-for="v in scanResult.videos" :key="v" class="scan-item">🎬 {{ baseName(v) }}</div>
    </div>
    <template #footer>
      <el-button @click="confirmVisible = false">取消</el-button>
      <el-button type="primary" @click="startBatch">开始批量翻译</el-button>
    </template>
  </el-dialog>

  <el-dialog
    v-model="browserVisible"
    :title="browsePick === 'dir' ? '选择目录' : '选择视频文件'"
    width="680px"
  >
    <div class="browser-path">
      <el-button size="small" :disabled="browser.parent === null" @click="openBrowser(browser.parent)">
        ↑ 上级
      </el-button>
      <el-input
        v-model="addressInput"
        size="small"
        placeholder="粘贴文件夹或视频文件的完整路径，回车跳转"
        @keyup.enter="jumpToAddress"
      >
        <template #append>
          <el-button @click="jumpToAddress">跳转</el-button>
        </template>
      </el-input>
    </div>
    <div v-if="quickAccess.length" class="quick-access">
      <el-tag
        v-for="q in quickAccess" :key="q.path"
        class="quick-item" effect="plain" @click="openBrowser(q.path)"
      >
        {{ q.name }}
      </el-tag>
    </div>
    <div class="browser-list">
      <div v-for="d in browser.dirs" :key="'d-' + d" class="entry dir" @click="openBrowser(browser.path ? joinPath(browser.path, d) : d)">
        📁 {{ d }}
      </div>
      <div v-for="f in browser.files" :key="'f-' + f.name" class="entry file" @click="pickFile(f.name)">
        🎬 {{ f.name }} <span class="size">{{ fmtSize(f.size) }}</span>
      </div>
      <el-empty v-if="!browser.dirs.length && !browser.files.length" description="此目录没有子目录或视频文件" :image-size="60" />
    </div>
    <template v-if="browsePick === 'dir'" #footer>
      <el-button @click="browserVisible = false">取消</el-button>
      <el-button type="primary" :disabled="!browser.path" @click="pickThisDir">
        ✓ 选择此目录
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.hint {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
.progress-card {
  margin-top: 16px;
}
.progress-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.message {
  color: #606266;
  font-size: 13px;
}
.logs {
  height: 220px;
  overflow-y: auto;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
  padding: 8px 12px;
  border-radius: 6px;
  margin-bottom: 12px;
  white-space: pre-wrap;
  word-break: break-all;
}
.browser-path {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.quick-access {
  margin-bottom: 8px;
}
.batch-files {
  max-height: 220px;
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  margin-bottom: 12px;
}
.batch-file {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 5px 12px;
  border-bottom: 1px solid #f5f7fa;
  font-size: 13px;
}
.batch-file .name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.scan-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 4px 12px;
  font-size: 13px;
  line-height: 1.9;
}
.quick-item {
  margin: 0 6px 4px 0;
  cursor: pointer;
}
.browser-list {
  max-height: 380px;
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 6px;
}
.entry {
  padding: 7px 12px;
  cursor: pointer;
  border-bottom: 1px solid #f5f7fa;
}
.entry:hover {
  background: #ecf5ff;
}
.entry .size {
  float: right;
  color: #909399;
  font-size: 12px;
}
</style>

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

const form = reactive({
  video_path: '',
  source_language: 'auto',
  target_language: '简体中文',
  synopsis: '',
  output_mode: 'bilingual',
})

// ---------------------------------------------------------- file browser
const browserVisible = ref(false)
const browser = reactive({ path: '', parent: null, dirs: [], files: [] })

async function openBrowser(path = '') {
  try {
    const data = await api.browse(path)
    Object.assign(browser, data)
    browserVisible.value = true
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
  form.video_path = joinPath(browser.path, name)
  browserVisible.value = false
}

function fmtSize(bytes) {
  if (bytes > 1 << 30) return (bytes / (1 << 30)).toFixed(1) + ' GB'
  if (bytes > 1 << 20) return (bytes / (1 << 20)).toFixed(1) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

// -------------------------------------------------------- drag & drop
const uploading = ref(null) // { name, progress }

function onDrop(e) {
  const f = e.dataTransfer?.files?.[0]
  if (!f) return
  uploadFile(f)
}

function uploadFile(f) {
  const fd = new FormData()
  fd.append('file', f)
  const xhr = new XMLHttpRequest()
  xhr.open('POST', '/api/upload')
  uploading.value = { name: f.name, progress: 0 }
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable && uploading.value)
      uploading.value.progress = Math.round((ev.loaded / ev.total) * 100)
  }
  xhr.onload = () => {
    uploading.value = null
    if (xhr.status === 200) {
      form.video_path = JSON.parse(xhr.responseText).path
      ElMessage.success('已添加: ' + f.name)
    } else {
      let msg = xhr.statusText
      try { msg = JSON.parse(xhr.responseText).detail || msg } catch { /* keep */ }
      ElMessage.error('添加失败: ' + msg)
    }
  }
  xhr.onerror = () => {
    uploading.value = null
    ElMessage.error('上传中断，请重试或使用「浏览」按钮')
  }
  xhr.send(fd)
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
      if (ev.stage === 'done') ElMessage.success('字幕生成完成！')
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

onBeforeUnmount(() => eventSource?.close())
</script>

<template>
  <el-card shadow="never">
    <el-form :model="form" label-width="110px">
      <el-form-item label="视频文件" required>
        <div style="width: 100%">
          <el-input v-model="form.video_path" placeholder="视频文件的完整路径">
            <template #append>
              <el-button @click="openBrowser()">浏览…</el-button>
            </template>
          </el-input>
          <div
            class="dropzone"
            @dragover.prevent
            @dragenter.prevent
            @drop.prevent="onDrop"
          >
            <template v-if="uploading">
              正在添加 {{ uploading.name }}…
              <el-progress :percentage="uploading.progress" style="margin-top: 4px" />
            </template>
            <template v-else>⬇ 或将视频文件拖拽到此处</template>
          </div>
        </div>
      </el-form-item>
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
        <el-button type="primary" :disabled="running()" @click="start">开始翻译</el-button>
        <el-button v-if="running()" type="danger" plain @click="cancel">取消任务</el-button>
      </el-form-item>
    </el-form>
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
    <el-button
      v-if="job.stage === 'done'"
      type="success"
      tag="a"
      :href="api.resultUrl(job.id)"
      download
    >
      下载 SRT 字幕
    </el-button>
  </el-card>

  <el-dialog v-model="browserVisible" title="选择视频文件" width="620px">
    <div class="browser-path">
      <el-button size="small" :disabled="browser.parent === null" @click="openBrowser(browser.parent)">
        ↑ 上级
      </el-button>
      <code>{{ browser.path || '磁盘列表' }}</code>
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
  </el-dialog>
</template>

<style scoped>
.hint {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
.dropzone {
  margin-top: 8px;
  padding: 14px;
  border: 2px dashed #c0c4cc;
  border-radius: 6px;
  text-align: center;
  color: #909399;
  font-size: 13px;
  transition: border-color 0.2s;
}
.dropzone:hover {
  border-color: #409eff;
  color: #409eff;
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

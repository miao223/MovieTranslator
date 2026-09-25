<script setup>
// 压制：把视频重新编码（CPU 或显卡），只走列队。
//
// Two ways in, like the 原盘 page: one file, or a folder whose videos are
// all listed (discs inside it are left to the 原盘 page). The page asks the
// server what a file holds (/api/encode/probe, /api/encode/scan), shows it,
// and adds the encode to the queue — optionally with the subtitles made
// once it is written. It never deletes or overwrites anything: the encode
// goes next to its source under a new name (片名.HEVC.mkv), or into a
// chosen folder under the source's own.
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import FileBrowser from '../components/FileBrowser.vue'
import EncodeFields from '../components/EncodeFields.vue'
import SubtitleDialog from '../components/SubtitleDialog.vue'
import { defaultEncodeOptions, encodedName, encoderCaps } from '../encode'

const emit = defineEmits(['goto'])

const mode = ref('single')          // 'single' | 'batch'
const opts = ref(defaultEncodeOptions())
const outputMode = ref('beside')    // 'beside' | 'custom'
const customDir = ref('')
const encoders = ref([])

onMounted(async () => {
  try {
    const s = await api.getSettings()
    if (s.encode) opts.value = { ...s.encode }
  } catch {
    /* the built-in defaults */
  }
  try {
    encoders.value = (await encoderCaps()).encoders
  } catch {
    /* EncodeFields says so */
  }
})

// ------------------------------------------------------------ one file
const path = ref('')
const info = ref(null)
const probing = ref(false)
const probeError = ref('')
let probeTimer = null

watch(path, () => {
  info.value = null
  probeError.value = ''
  enqueued.value = null
  clearTimeout(probeTimer)
  if (path.value.trim()) probeTimer = setTimeout(probe, 400)
})

async function probe() {
  const target = path.value.trim()
  if (!target) return
  probing.value = true
  try {
    info.value = await api.encodeProbe(target)
    probeError.value = ''
  } catch (e) {
    info.value = null
    probeError.value = e.message
  } finally {
    probing.value = false
  }
}

// ------------------------------------------------------- a whole folder
const folder = ref('')
const recursive = ref(true)
const scan = ref(null)
const scanning = ref(false)
const scanError = ref('')
const ticked = reactive({})

async function scanFolder() {
  const target = folder.value.trim()
  if (!target) return
  scanning.value = true
  scanError.value = ''
  enqueued.value = null
  try {
    const r = await api.encodeScan({ path: target, recursive: recursive.value })
    scan.value = r
    for (const k of Object.keys(ticked)) delete ticked[k]
    // a file that is already one of ours is listed, not ticked
    for (const f of r.files) ticked[f.path] = !f.encoded
  } catch (e) {
    scan.value = null
    scanError.value = e.message
  } finally {
    scanning.value = false
  }
}

watch(folder, () => {
  scan.value = null
  scanError.value = ''
})
watch(recursive, () => { if (scan.value) scanFolder() })

const files = computed(() => scan.value?.files || [])
const picked = computed(() => files.value.filter((f) => ticked[f.path]))
const pickedBytes = computed(() => picked.value.reduce((n, f) => n + f.size, 0))
const allTicked = computed(() => files.value.length > 0 && picked.value.length === files.value.length)

function tickAll(value) {
  for (const f of files.value) ticked[f.path] = value
}

// ---------------------------------------------------------- the output
function stemOf(name) {
  return name.replace(/\.[^.]+$/, '')
}

function dirOf(p) {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i > 0 ? p.slice(0, i) : p
}

const custom = computed(() => outputMode.value === 'custom')
const outName = computed(() => (info.value
  ? encodedName(stemOf(info.value.name), encoders.value, opts.value, custom.value) : ''))
const outFolder = computed(() => (custom.value ? customDir.value.trim()
  : info.value ? dirOf(info.value.path) : ''))

// ------------------------------------------------------------ queueing
const busy = ref(false)
const enqueued = ref(null)          // { position, count, subtitles }
const subDialog = ref(false)

const canQueue = computed(() => !busy.value
  && (!custom.value || !!customDir.value.trim())
  && (mode.value === 'single' ? !!info.value : picked.value.length > 0))

// subtitles: the 「加入列队并做字幕」 dialog's answer, null for plain 加入列队
async function enqueue(subtitles = null) {
  busy.value = true
  try {
    const out = { output_mode: outputMode.value, output_dir: custom.value ? customDir.value.trim() : '' }
    const resp = mode.value === 'single'
      ? await api.enqueueEncode({ source: info.value.path, options: opts.value, ...out, subtitles })
      : await api.enqueueEncodeBatch({
        path: scan.value.path, files: picked.value.map((f) => f.path),
        options: opts.value, ...out, subtitles,
      })
    enqueued.value = { position: resp.position, count: resp.count, subtitles: !!subtitles }
    subDialog.value = false
    ElMessage.success(resp.count > 1 ? `已加入列队：${resp.count} 个文件` : `已加入列队（第 ${resp.position} 位）`)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    busy.value = false
  }
}

// what the subtitle dialog is about: the tracks of the file(s) picked
const subjects = computed(() => (mode.value === 'single' ? (info.value ? [info.value] : []) : picked.value))

function languagesOf(kind) {
  const seen = new Set()
  for (const f of subjects.value) {
    for (const t of (kind === 'audio' ? f.audio : f.subtitles)) seen.add(t.language_name || '未标注')
  }
  return [...seen]
}

// ------------------------------------------------------------- browsing
const browserVisible = ref(false)
const browseMode = ref('video')
const browseFor = ref('path')       // 'path' | 'folder' | 'output'
const fileBrowser = ref(null)

function openBrowser(target) {
  browseFor.value = target
  browseMode.value = target === 'path' ? 'video' : 'dir'
  const start = target === 'output' ? customDir.value
    : target === 'folder' ? folder.value : (path.value ? dirOf(path.value) : '')
  setTimeout(() => fileBrowser.value?.open(start), 0)
}

function onPick(p) {
  if (browseFor.value === 'output') customDir.value = p
  else if (browseFor.value === 'folder') folder.value = p
  else path.value = p
}

// ------------------------------------------------------------ showing
function fmtBytes(bytes) {
  if (bytes == null) return '—'
  if (bytes >= 1 << 30) return (bytes / (1 << 30)).toFixed(2) + ' GB'
  if (bytes >= 1 << 20) return (bytes / (1 << 20)).toFixed(0) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

function fmtDuration(seconds) {
  const s = Math.round(seconds || 0)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

const CHANNELS = { 1: '单声道', 2: '立体声', 6: '5.1', 8: '7.1' }

// ffmpeg's demuxer names ("matroska,webm", "mov,mp4,m4a,3gp,3g2,mj2") are
// not what anyone calls these files
function containerName(name) {
  const known = [['matroska', 'MKV'], ['mp4', 'MP4'], ['mpegts', 'TS'], ['avi', 'AVI'],
    ['flv', 'FLV'], ['asf', 'WMV'], ['mpeg', 'MPEG'], ['ogg', 'OGG']]
  return known.find(([key]) => name.includes(key))?.[1] || name
}

function videoLine(v) {
  const parts = [`${v.codec.toUpperCase()} ${v.width}×${v.height}`, `${v.fps} fps`, `${v.bit_depth}bit`]
  if (v.sar) parts.push(`像素比例 ${v.sar}`)
  return parts.join(' · ')
}

function audioLine(a) {
  return [a.language_name, a.profile || a.codec.toUpperCase(),
    CHANNELS[a.channels] || `${a.channels} 声道`].join(' ')
    + (a.title ? `「${a.title}」` : '')
}
</script>

<template>
  <el-card shadow="never">
    <el-form label-width="110px" @submit.prevent>
      <el-form-item label="方式">
        <el-radio-group v-model="mode">
          <el-radio value="single">单个视频</el-radio>
          <el-radio value="batch">批量（一个文件夹里的所有视频）</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item v-if="mode === 'single'" label="视频" required>
        <el-input v-model="path" placeholder="要压制的视频文件的完整路径">
          <template #append>
            <el-button @click="openBrowser('path')">浏览…</el-button>
          </template>
        </el-input>
      </el-form-item>
      <template v-if="mode === 'batch'">
        <el-form-item label="文件夹" required>
          <el-input
            v-model="folder" placeholder="放着视频的文件夹：里面的视频会全部列出来"
            @keyup.enter="scanFolder"
          >
            <template #append>
              <el-button @click="openBrowser('folder')">浏览…</el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item label="包含子文件夹">
          <el-switch v-model="recursive" />
        </el-form-item>
      </template>
      <el-form-item label="输出位置">
        <el-radio-group v-model="outputMode">
          <el-radio value="beside">放在原文件旁边</el-radio>
          <el-radio value="custom">全部放到指定文件夹</el-radio>
        </el-radio-group>
        <el-input
          v-if="custom" v-model="customDir" class="custom-dir" placeholder="输出文件夹的完整路径"
        >
          <template #append>
            <el-button @click="openBrowser('output')">浏览…</el-button>
          </template>
        </el-input>
        <div class="hint below">
          <template v-if="!custom">
            文件名加上编码：<code>Film.mkv</code> 旁边写出 <code>Film.HEVC.mkv</code>。原文件不动。
          </template>
          <template v-else>
            保持原来的文件名；批量时每个文件在指定文件夹里保留原来的子文件夹。已有同名文件时另起名，绝不覆盖。
          </template>
        </div>
      </el-form-item>
      <el-form-item v-if="mode === 'batch'">
        <el-button type="primary" :loading="scanning" :disabled="!folder.trim()" @click="scanFolder">
          列出文件夹里的视频
        </el-button>
      </el-form-item>
      <div class="hint below intro">
        把视频<strong>重新编码</strong>（压制）成更小的文件：H.265 / AV1 / H.264 / VP9，CPU 或显卡都行；
        章节、字幕轨、字体附件和各音轨的语言原样带上。压制很慢，会一直占着列队（同一时间只跑一个任务），
        期间别的任务都在排队。
      </div>
    </el-form>
    <el-alert v-if="mode === 'single' && probeError" type="error" :closable="false" :title="probeError" />
    <el-alert v-if="mode === 'batch' && scanError" type="error" :closable="false" :title="scanError" />
  </el-card>

  <!-- ============================================================ one file -->
  <el-card v-if="mode === 'single' && (info || probing)" shadow="never" class="block" v-loading="probing">
    <template v-if="info">
      <div class="file-head">
        <span class="file-name">🎞️ {{ info.name }}</span>
        <el-tag size="small" type="info">{{ containerName(info.container) }}</el-tag>
        <span class="file-meta">{{ fmtDuration(info.duration) }} · {{ fmtBytes(info.size) }}</span>
        <el-tag v-if="info.encoded" size="small" type="warning">已经是本程序压制过的文件</el-tag>
      </div>
      <div class="tracks">
        <div v-if="info.video">
          🎬 {{ videoLine(info.video) }}
          <el-tag v-if="info.video.hdr" size="small" type="warning">{{ info.video.hdr }}</el-tag>
          <el-tag v-if="info.video.interlaced >= 0.5" size="small" type="info">隔行</el-tag>
        </div>
        <div v-for="a in info.audio" :key="a.index">
          🔊 {{ audioLine(a) }}
          <el-tag v-if="a.lossless" size="small" type="success">无损</el-tag>
          <el-tag v-if="a.atmos" size="small" type="info">声音对象（Atmos / DTS:X）</el-tag>
        </div>
        <div v-if="info.subtitles.length">
          💬 字幕 {{ info.subtitles.length }} 条：{{ info.subtitles.map((s) => s.language_name
            + (s.bitmap ? '（图形）' : '')).join('、') }}
        </div>
        <div v-if="info.chapters || info.attachments" class="file-meta">
          <template v-if="info.chapters">章节 {{ info.chapters }} 个</template>
          <template v-if="info.chapters && info.attachments"> · </template>
          <template v-if="info.attachments">附件 {{ info.attachments }} 个（字体等）</template>
        </div>
      </div>
    </template>
  </el-card>

  <!-- ======================================================= a whole folder -->
  <el-card v-if="mode === 'batch' && scan" shadow="never" class="block">
    <div class="file-head">
      <span class="file-name">📁 找到 {{ files.length }} 个视频</span>
      <code class="file-meta">{{ scan.path }}</code>
    </div>
    <el-alert
      v-if="scan.discs.length" type="info" :closable="false" show-icon style="margin: 8px 0"
      :title="`文件夹里还有 ${scan.discs.length} 张原盘（没有列进来）：原盘请到「原盘」页封装，那里也能一路压制、做字幕`"
    />
    <el-table v-if="files.length" :data="files" size="small" class="files" max-height="420">
      <el-table-column width="46">
        <template #header>
          <el-checkbox :model-value="allTicked" @update:model-value="tickAll" />
        </template>
        <template #default="{ row }">
          <el-checkbox :model-value="!!ticked[row.path]" @update:model-value="ticked[row.path] = $event" />
        </template>
      </el-table-column>
      <el-table-column label="文件" min-width="260">
        <template #default="{ row }">
          <span class="rel">{{ row.relative }}</span>
          <el-tag v-if="row.encoded" size="small" type="warning" class="tag">已压制</el-tag>
          <el-tag v-if="row.video?.hdr" size="small" type="warning" class="tag">{{ row.video.hdr }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="画面" width="190">
        <template #default="{ row }">
          {{ row.video.codec.toUpperCase() }} {{ row.video.width }}×{{ row.video.height }}
        </template>
      </el-table-column>
      <el-table-column label="时长" width="84">
        <template #default="{ row }">{{ fmtDuration(row.duration) }}</template>
      </el-table-column>
      <el-table-column label="大小" width="90" align="right">
        <template #default="{ row }">{{ fmtBytes(row.size) }}</template>
      </el-table-column>
    </el-table>
    <div v-if="scan.skipped.length" class="skipped">
      <div class="hint">以下 {{ scan.skipped.length }} 个读不出来，没有列进来：</div>
      <div v-for="s in scan.skipped" :key="s.path" class="reason">✕ <code>{{ s.path }}</code>：{{ s.reason }}</div>
    </div>
  </el-card>

  <!-- ========================================================== the options -->
  <el-card shadow="never" class="block">
    <template #header>
      <span class="card-title">压制参数</span>
      <span class="hint">默认值在「设置 → 视频压制」里改；这里改的只对这次有效</span>
    </template>
    <EncodeFields v-model="opts" :source="mode === 'single' ? info : null" />
  </el-card>

  <el-card shadow="never" class="block">
    <div class="footer">
      <div class="summary">
        <template v-if="mode === 'single'">
          <template v-if="info">
            → <code>{{ outName }}</code>
            <template v-if="outFolder">，写到 <code>{{ outFolder }}</code></template>
          </template>
          <template v-else>先选一个视频</template>
        </template>
        <template v-else>
          已选 <strong>{{ picked.length }}</strong> 个文件，共 <strong>{{ fmtBytes(pickedBytes) }}</strong>
        </template>
      </div>
      <div class="buttons">
        <el-button type="primary" :disabled="!canQueue" :loading="busy" @click="enqueue()">
          ＋ 加入列队{{ mode === 'batch' && picked.length ? `（${picked.length} 个）` : '' }}
        </el-button>
        <el-button type="primary" plain :disabled="!canQueue" @click="subDialog = true">
          ＋ 加入列队并做字幕
        </el-button>
      </div>
    </div>
    <el-alert v-if="enqueued" type="success" :closable="false" style="margin-top: 10px">
      <template #title>
        <template v-if="enqueued.count > 1">
          已加入列队：{{ enqueued.count }} 个文件，每个一条任务（从第 {{ enqueued.position }} 位起）。
        </template>
        <template v-else>已加入列队（第 {{ enqueued.position }} 位）。</template>
        <template v-if="enqueued.subtitles">压制完成后，会自动加入字幕任务（排在列队最后）。</template>
        <el-link type="primary" :underline="false" @click="emit('goto', 'queue')">去列队查看进度 →</el-link>
      </template>
    </el-alert>
  </el-card>

  <FileBrowser ref="fileBrowser" v-model="browserVisible" :mode="browseMode" @pick="onPick" />
  <SubtitleDialog
    v-model="subDialog" :busy="busy" :count="mode === 'single' ? 1 : picked.length"
    after="压制" :series-scope="mode === 'single' ? 'none' : 'batch'"
    :audio="languagesOf('audio')" :subtitles="languagesOf('subtitle')"
    @confirm="enqueue"
  />
</template>

<style scoped>
.hint {
  margin-left: 12px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.hint.below {
  display: block;
  width: 100%;
  margin: 4px 0 0;
  line-height: 1.6;
}
.hint.intro {
  margin: 0 0 4px;
}
.custom-dir {
  margin-top: 6px;
  max-width: 560px;
}
.block {
  margin-top: 16px;
}
.card-title {
  font-weight: 600;
}
.file-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.file-name {
  font-weight: 600;
  font-size: 15px;
  word-break: break-all;
}
.file-meta {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  word-break: break-all;
}
.tracks {
  margin-top: 10px;
  font-size: 13px;
  line-height: 1.9;
  color: var(--el-text-color-regular);
}
.files {
  margin-top: 8px;
}
.rel {
  word-break: break-all;
}
.tag {
  margin-left: 6px;
}
.skipped {
  margin-top: 12px;
}
.reason {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.6;
}
.footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.summary {
  font-size: 13px;
  color: var(--el-text-color-regular);
  word-break: break-all;
}
.buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
</style>

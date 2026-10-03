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
//
// The 修复 page is this page with `restore` set: a restore is an encode
// with more done to the picture first (services/restore.py), so it shares
// everything here and adds the RestoreFields card.
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import FileBrowser from '../components/FileBrowser.vue'
import EncodeFields from '../components/EncodeFields.vue'
import RestoreFields from '../components/RestoreFields.vue'
import SubtitleDialog from '../components/SubtitleDialog.vue'
import { defaultEncodeOptions, encodedName, encoderCaps, RESTORE_OFF } from '../encode'

const props = defineProps({
  restore: { type: Boolean, default: false },
})
const emit = defineEmits(['goto'])

const mode = ref('single')          // 'single' | 'batch'
const opts = ref(defaultEncodeOptions())
const outputMode = ref('beside')    // 'beside' | 'custom'
const customDir = ref('')
const encoders = ref([])
const engineReady = ref(false)

onMounted(async () => {
  try {
    const s = await api.getSettings()
    if (s.encode) opts.value = { ...s.encode }
    engineReady.value = !!s.restore?.engine_python
  } catch {
    /* the built-in defaults */
  }
  if (props.restore) {
    // what the picture is decides the deinterlacing (a folder of mixed discs
    // gets each one right), and black bars go before anything is enlarged
    opts.value = { ...opts.value, ...RESTORE_OFF, deinterlace: 'detect', crop_auto: true, max_height: 0 }
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
  analysis.value = null
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
  if (props.restore && info.value) analyze(info.value.path)
}

// ------------------------------------------ 修复: what the picture is
//
// The same measurement 自动判断 makes when the encode starts
// (restore.resolve), shown here first so the user sees what it will do.
const analysis = ref(null)
const analyzing = ref(false)

async function analyze(file) {
  analyzing.value = true
  try {
    const r = await api.restoreAnalyze(file)
    if (info.value?.path === file) analysis.value = r
  } catch (e) {
    if (info.value?.path === file) ElMessage.warning(`分析不了这个文件：${e.message}`)
  } finally {
    analyzing.value = false
  }
}

// ---------------------------------------------- 试看 (修复, one file)
//
// A few moments of this film through Lanczos and every AI model on this
// machine's GPU (services/restore_preview.py), side by side, with each
// model's measured speed. It never picks the model: the user's choice was
// three models, no default — only the 用这个 buttons set opts.ai_model.
const preview = ref(null)           // the server's view of the current preview
const previewStarting = ref(false)
let previewTimer = null
const previewRunning = computed(() => ['waiting', 'running'].includes(preview.value?.status))
const previewForThis = computed(() => !!(preview.value && info.value
  && preview.value.source === info.value.path))
const canPreview = computed(() => !!info.value && !!opts.value.upscale && engineReady.value
  && !previewStarting.value && !(previewRunning.value && previewForThis.value))
// the form moved on since: the segments were run with something else
const PREVIEW_KEYS = ['deinterlace', 'field_order', 'crop_top', 'crop_bottom', 'crop_left',
  'crop_right', 'crop_auto', 'aspect', 'denoise', 'upscale']
const previewStale = computed(() => previewForThis.value && preview.value.status === 'done'
  && PREVIEW_KEYS.some((k) => preview.value.options[k] !== opts.value[k]))

const METHOD_NAMES = {
  lanczos: '原片（Lanczos）', realbasicvsr: 'RealBasicVSR', realviformer: 'RealViformer',
  liveaction_span: '2xLiveActionV1_SPAN',
}
function fmtHours(h) {
  if (h == null) return '—'
  return h < 1 ? `${Math.max(1, Math.round(h * 60))} 分钟` : `${h.toFixed(1)} 小时`
}
function fmtTime(seconds) {
  const s = Math.floor(seconds)
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`
}
const previewRows = computed(() => {
  const methods = preview.value?.methods || {}
  const base = methods.lanczos?.flicker
  return Object.keys(METHOD_NAMES).filter((m) => methods[m]).map((m) => {
    const f = methods[m]
    return {
      method: m,
      name: METHOD_NAMES[m],
      speed: f.model_fps ? `${f.model_fps.toFixed(1)} 帧/秒` : '—',
      model: fmtHours(f.film_model_hours),
      total: fmtHours(f.film_total_hours),
      vram: f.peak_reserved_gb ? `${f.peak_reserved_gb.toFixed(1)} GB` : '—',
      flicker: f.flicker == null ? '—'
        : `${f.flicker.toFixed(2)}${base && m !== 'lanczos' ? `（${(f.flicker / base).toFixed(1)}×）` : ''}`,
    }
  })
})
const previewVideo = computed(() => (preview.value?.files?.video
  ? api.restorePreviewFile(preview.value.files.video, preview.value.id) : ''))
const previewSheet = computed(() => (preview.value?.files?.sheet
  ? api.restorePreviewFile(preview.value.files.sheet, preview.value.id) : ''))
const previewWhere = computed(() => (preview.value?.segments || [])
  .map(([start]) => fmtTime(start)).join('、'))

function chosen(method) {
  return (opts.value.ai_model || 'lanczos') === method
}
function useMethod(method) {
  opts.value.ai_model = method === 'lanczos' ? '' : method
}

async function startPreview() {
  previewStarting.value = true
  try {
    preview.value = await api.restorePreview(info.value.path, { ...opts.value })
    pollPreview()
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    previewStarting.value = false
  }
}
async function cancelPreview() {
  try {
    preview.value = await api.restorePreviewCancel()
  } catch (e) {
    ElMessage.error(e.message)
  }
}
function pollPreview() {
  clearTimeout(previewTimer)
  previewTimer = setTimeout(async () => {
    try {
      preview.value = await api.restorePreviewStatus()
    } catch {
      /* the next round */
    }
    if (previewRunning.value) pollPreview()
  }, 2000)
}
onMounted(async () => {
  if (!props.restore) return
  try {
    const p = await api.restorePreviewStatus()
    if (p.status !== 'idle') preview.value = p
    if (previewRunning.value) pollPreview()
  } catch {
    /* no preview to show */
  }
})
onBeforeUnmount(() => clearTimeout(previewTimer))

// ----------------------------------------------- 按画面选, for one file
//
// Asked as soon as the file is known, so the choice is in the form before
// it is queued — and can still be changed there. What is queued is the
// form as it stands (auto_pick off), never a second judgement at run time.
// A folder and the 原盘 page's full pipeline judge each file as it starts.
const pick = ref(null)              // /api/encode/pick's answer, for pick.path
const picking = ref(false)
const pickError = ref('')
const PICKED = ['video_codec', 'preset', 'quality', 'tune', 'rate_control']

async function runPick() {
  const file = info.value?.path
  if (!file) return
  picking.value = true
  pickError.value = ''
  try {
    const r = await api.encodePick({ path: file, options: opts.value })
    if (info.value?.path !== file) return      // another file was chosen meanwhile
    for (const k of PICKED) opts.value[k] = r.options[k]
    pick.value = { ...r, path: file }
  } catch (e) {
    pick.value = null
    pickError.value = e.message
  } finally {
    picking.value = false
  }
}

watch([info, () => opts.value.auto_pick], () => {
  if (mode.value !== 'single' || !info.value) return
  if (!opts.value.auto_pick) {
    pickError.value = ''
    return
  }
  if (pick.value?.path !== info.value.path && !picking.value) runPick()
})

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
  ? encodedName(stemOf(info.value.name), encoders.value, opts.value, custom.value,
    info.value.video?.fps, analysis.value) : ''))
const outFolder = computed(() => (custom.value ? customDir.value.trim()
  : info.value ? dirOf(info.value.path) : ''))

// ------------------------------------------------------------ queueing
const busy = ref(false)
const enqueued = ref(null)          // { position, count, subtitles }
const subDialog = ref(false)

const canQueue = computed(() => !busy.value && !picking.value
  && (!custom.value || !!customDir.value.trim())
  && (mode.value === 'single' ? !!info.value : picked.value.length > 0))

// subtitles: the 「加入列队并做字幕」 dialog's answer, null for plain 加入列队
async function enqueue(subtitles = null) {
  busy.value = true
  try {
    const out = { output_mode: outputMode.value, output_dir: custom.value ? customDir.value.trim() : '' }
    const resp = mode.value === 'single'
      // the form already holds what the picture was judged to need
      ? await api.enqueueEncode({ source: info.value.path, options: { ...opts.value, auto_pick: false },
        ...out, subtitles })
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
        <el-input v-model="path" :placeholder="restore ? '要修复的视频文件的完整路径' : '要压制的视频文件的完整路径'">
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
            <template v-if="restore">
              文件名加上做了什么：<code>Film.mkv</code> 旁边写出 <code>Film.1080p.60fps.HEVC.mkv</code>
              （放大了才有 1080p，还原 60 帧才有 60fps）。原文件不动。
            </template>
            <template v-else>
              文件名加上编码：<code>Film.mkv</code> 旁边写出 <code>Film.HEVC.mkv</code>。原文件不动。
            </template>
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
      <div v-if="restore" class="hint below intro">
        给老的 <strong>VHS 录像、DVD</strong> 提升画质：隔行录像还原成每秒 60 帧、裁掉磁头杂波和黑边、降噪、放大，
        然后重新编码。原文件不动。和压制一样只走列队、同一时间只跑一个任务。
      </div>
      <div v-else class="hint below intro">
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
      <div v-if="opts.auto_pick" class="pick" v-loading="picking">
        <template v-if="picking">🤖 视觉模型正在看画面…</template>
        <template v-else-if="pick && pick.path === info.path">
          <el-image
            :src="pick.image" :preview-src-list="[pick.image]" fit="cover" class="pick-sheet"
            title="模型看到的拼图：上两行 6 个时间点的整帧，最下一行 1:1 原始像素"
          />
          <div>
            🤖 视觉模型判断：<strong>{{ pick.summary }}</strong>
            <template v-if="pick.reason">（{{ pick.reason }}）</template>
            <div class="file-meta">已填进下面的压制参数：{{ pick.described }}。可以再改。</div>
          </div>
        </template>
        <el-alert v-else-if="pickError" type="warning" :closable="false" show-icon>
          <template #title>
            {{ pickError }}；用下面的参数。
            <el-link type="primary" :underline="false" @click="runPick">重新判断</el-link>
          </template>
        </el-alert>
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
  <el-card v-if="restore" shadow="never" class="block">
    <template #header>
      <span class="card-title">修复参数</span>
      <span class="hint">依次做：反交错 → 裁边 → 降噪 → 放大 → 编码</span>
    </template>
    <RestoreFields
      v-model="opts" :source="mode === 'single' ? info : null"
      :analysis="mode === 'single' ? analysis : null" :analyzing="mode === 'single' && analyzing"
      :engine-ready="engineReady"
    />
  </el-card>

  <el-card v-if="restore && mode === 'single'" shadow="never" class="block">
    <template #header>
      <span class="card-title">试看</span>
      <span class="hint">从这部片里取 3 段、各 3 秒，用原片（Lanczos）和三个模型各跑一遍，并排比较；胶片片源约 5 分钟，还原 60 帧的录像约 10–15 分钟</span>
    </template>
    <div v-if="!info" class="hint intro">先在上面选一个视频。</div>
    <template v-else>
      <div class="preview-actions">
        <el-button type="primary" plain :disabled="!canPreview" :loading="previewStarting" @click="startPreview">
          {{ previewForThis && preview.status === 'done' ? '重新试看' : '开始试看' }}
        </el-button>
        <el-button v-if="previewRunning" @click="cancelPreview">取消试看</el-button>
        <span v-if="!opts.upscale" class="hint">先在「修复参数」里选「放大到」：比的是放大的方法</span>
        <span v-else-if="!engineReady" class="hint">AI 模型要先在「设置 → 修复引擎」里配置引擎</span>
        <span v-else class="hint">用上面的修复参数；和翻译、压制轮流用显卡，正在跑别的任务时会先等它</span>
      </div>
      <div v-if="preview && previewRunning && !previewForThis" class="hint below">
        正在试看另一个文件（{{ preview.source }}）；在这里开始会取消它。
      </div>
      <template v-if="previewForThis">
        <div v-if="previewRunning" class="preview-progress">
          <el-progress :percentage="Math.round(preview.progress * 100)" />
          <div class="hint below">{{ preview.stage }}</div>
        </div>
        <el-alert
          v-else-if="preview.status === 'failed'" type="error" :closable="false" show-icon
          :title="`试看失败：${preview.error}`" class="preview-alert"
        />
        <el-alert
          v-else-if="preview.status === 'cancelled'" type="info" :closable="false"
          title="试看已取消" class="preview-alert"
        />
        <template v-else-if="preview.status === 'done'">
          <el-alert
            v-if="previewStale" type="warning" :closable="false" show-icon class="preview-alert"
            title="修复参数改过了：下面是按改之前的参数跑的，想看现在的效果请重新试看"
          />
          <div class="hint below">
            片段：{{ previewWhere }}（每段 3 秒）。用的是：{{ preview.resolved.join('，') }}。
          </div>
          <video
            :src="previewVideo" class="preview-video" controls loop muted playsinline preload="metadata"
          />
          <div class="hint below">
            左上 原片（Lanczos）｜右上 RealBasicVSR｜左下 RealViformer｜右下 SPAN，三段接在一起，每一格都是原尺寸。
            闪烁要看动起来的画面；卡的话 <a :href="previewVideo" download="试看.mp4">下载</a> 下来用播放器看。
          </div>
          <el-table :data="previewRows" size="small" class="preview-table">
            <el-table-column prop="name" label="方法" min-width="150" />
            <el-table-column prop="speed" label="模型速度" width="100" />
            <el-table-column prop="model" label="这部片：只算模型" width="130" />
            <el-table-column prop="total" label="含解码和编码" width="120" />
            <el-table-column prop="vram" label="显存" width="80" />
            <el-table-column prop="flicker" label="闪烁（× 原片）" width="130" />
            <el-table-column label="" width="96" align="center">
              <template #default="{ row }">
                <el-tag v-if="chosen(row.method)" type="success" size="small">已选</el-tag>
                <el-button
                  v-else size="small" :disabled="row.method !== 'lanczos' && !engineReady"
                  @click="useMethod(row.method)"
                >用这个</el-button>
              </template>
            </el-table-column>
          </el-table>
          <div class="hint below">
            时间按这台机器上量到的速度、整部片的长度估算；「含解码和编码」用的是 x264 fast，换成更慢的编码器会更久。
            闪烁是静止区域里相邻两帧的平均差，原片是这部片自己的底。
          </div>
          <el-image :src="previewSheet" :preview-src-list="[previewSheet]" fit="contain" class="preview-sheet" />
          <div class="hint below">静帧：每段中间那一帧细节最多的地方，原尺寸；点开看大图。</div>
        </template>
      </template>
    </template>
  </el-card>

  <el-card shadow="never" class="block">
    <template #header>
      <span class="card-title">压制参数</span>
      <span class="hint">默认值在「设置 → 视频压制」里改；这里改的只对这次有效</span>
    </template>
    <EncodeFields
      v-model="opts" :source="mode === 'single' ? info : null" :restore="restore"
      :auto-hint="mode === 'single'
        ? '选好文件就先让视觉模型看一眼，判断结果填进下面的参数，加入列队前还能改。'
        : '每个文件开压前各让视觉模型看一次画面（每个文件一次调用）；判断不了时用下面这些参数。'"
    />
    <div class="advice">
      <div class="advice-title">想尽量保住画质，怎么选（实测：1080p 蓝光里颗粒重的电影、真人剧集、动画各一段，VMAF 打分；速度按 6 核 12 线程的 CPU）</div>
      <ul>
        <li>
          <strong>真人电影、剧集</strong>：H.265、速度「较慢」、CRF 18，和原片几乎看不出区别。颗粒是片子质感的（老胶片、颗粒明显的电影），「更多选项 → 内容类型」再选「保留胶片颗粒」：最接近原片，体积约翻倍。
        </li>
        <li>
          <strong>动画</strong>：AV1、速度「最慢」、CRF 25：分数最高，体积不到 H.265 的一半，还快 3 倍。颗粒或噪点重的老动画按真人片选。
        </li>
        <li>
          <strong>不建议</strong>：真人片用 AV1（会把颗粒当噪声抹掉，放大看皮肤发「塑料」，CRF 降到 18 也追不上）；H.264（除了颗粒极重的片子，同画质体积大得多）。
        </li>
        <li>
          <strong>要多久</strong>：H.265 较慢约 4–6 帧/秒，两小时的电影 8–12 小时，加「保留胶片颗粒」13–16 小时；嫌慢用「中等」+ CRF 18，快 2.6 倍、画质略低。AV1 最慢压动画约 25 帧/秒，一集 24 分钟约 25 分钟。
        </li>
      </ul>
      <div>打开上面的「按画面选」，就由视觉模型看截图、按这张表替你选。</div>
    </div>
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
        <template v-if="enqueued.subtitles">{{ restore ? "修复" : "压制" }}完成后，会自动加入字幕任务（排在列队最后）。</template>
        <el-link type="primary" :underline="false" @click="emit('goto', 'queue')">去列队查看进度 →</el-link>
      </template>
    </el-alert>
  </el-card>

  <FileBrowser ref="fileBrowser" v-model="browserVisible" :mode="browseMode" @pick="onPick" />
  <SubtitleDialog
    v-model="subDialog" :busy="busy" :count="mode === 'single' ? 1 : picked.length"
    :after="restore ? '修复' : '压制'" :series-scope="mode === 'single' ? 'none' : 'batch'"
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
.preview-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.preview-actions .hint {
  margin-left: 4px;
}
.preview-progress {
  margin-top: 12px;
  max-width: 640px;
}
.preview-alert {
  margin-top: 12px;
}
.preview-video {
  display: block;
  width: 100%;
  max-height: 70vh;
  margin-top: 12px;
  background: var(--el-color-black);
  border-radius: 4px;
}
.preview-table {
  margin-top: 12px;
}
.preview-sheet {
  display: block;
  width: 100%;
  margin-top: 12px;
}
.pick {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-top: 10px;
  min-height: 32px;
  font-size: 13px;
  line-height: 1.7;
  color: var(--el-text-color-regular);
}
.pick-sheet {
  width: 192px;
  height: 108px;
  flex: none;
  border-radius: var(--app-radius);
  cursor: zoom-in;
}
.advice {
  margin-top: 8px;
  padding: 10px 14px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--app-radius);
  font-size: 12px;
  line-height: 1.7;
  color: var(--el-text-color-secondary);
}
.advice-title {
  font-weight: 600;
  color: var(--el-text-color-regular);
}
.advice ul {
  margin: 4px 0;
  padding-left: 18px;
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

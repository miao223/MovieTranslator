<script setup>
// 音频：把视频的一条音轨提取成 16 kHz 单声道的 FLAC（无损，默认）或 Opus（小），只走列队。
//
// For translation: 16 kHz mono is what the pipeline turns every source into
// before transcribing. The FLAC decodes sample for sample to that WAV, so
// translating it is translating the video; Opus is a sixth of the size and
// the speech detection hears it very slightly differently.
// Laid out like the 压制 page — one file or a folder, the output beside
// the source or in a chosen folder, never over anything.
import { computed, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import FileBrowser from '../components/FileBrowser.vue'

const emit = defineEmits(['goto'])

// the server's defaults (audioextract.FORMATS). FLAC has no rate to choose;
// its sizes are estimated at the rate measured on a film (81.7 MB in 5194 s
// of 16 kHz mono), which varies with the material. Opus below 24 kbps
// starts to lose speech (16k: silero found 94.7% of it, 24k and up 97.6%).
const FORMATS = {
  flac: { suffix: '.flac', rates: [], default: 0, estimate: 126 },
  opus: { suffix: '.opus', rates: [16, 24, 32, 48], default: 24 },
}

const mode = ref('single')          // 'single' | 'batch'
const format = ref('flac')
const kbps = ref(FORMATS.flac.default)
const outputMode = ref('beside')    // 'beside' | 'custom'
const customDir = ref('')

watch(format, (f) => { kbps.value = FORMATS[f].default })

// ------------------------------------------------------------ one file
const path = ref('')
const info = ref(null)
const track = ref(null)
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
    info.value = await api.audioProbe(target)
    track.value = info.value.default_track
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
const language = ref('')            // '' = each file's default track

async function scanFolder() {
  const target = folder.value.trim()
  if (!target) return
  scanning.value = true
  scanError.value = ''
  enqueued.value = null
  try {
    const r = await api.audioScan({ path: target, recursive: recursive.value })
    scan.value = r
    for (const k of Object.keys(ticked)) delete ticked[k]
    // one already extracted is listed, not ticked
    for (const f of r.files) ticked[f.path] = !f.extracted
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
const allTicked = computed(() => files.value.length > 0 && picked.value.length === files.value.length)

function tickAll(value) {
  for (const f of files.value) ticked[f.path] = value
}

// the languages the ticked files have between them, for choosing by
const languages = computed(() => {
  const seen = new Map()
  for (const f of picked.value) {
    for (const t of f.tracks) {
      if (!t.language || t.language === 'und') continue
      const entry = seen.get(t.language) || { code: t.language, name: t.language_name, count: 0 }
      entry.count += 1
      seen.set(t.language, entry)
    }
  }
  return [...seen.values()]
})

// how many ticked files have no track in the chosen language
const withoutLanguage = computed(() => (language.value
  ? picked.value.filter((f) => !f.tracks.some((t) => t.language === language.value)).length : 0))

// ---------------------------------------------------------- the output
function stemOf(name) {
  return name.replace(/\.[^.]+$/, '')
}

function dirOf(p) {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i > 0 ? p.slice(0, i) : p
}

const custom = computed(() => outputMode.value === 'custom')
const outName = computed(() => (info.value ? stemOf(info.value.name) + FORMATS[format.value].suffix : ''))
const outFolder = computed(() => (custom.value ? customDir.value.trim()
  : info.value ? dirOf(info.value.path) : ''))

const seconds = computed(() => (mode.value === 'single'
  ? (info.value?.duration || 0)
  : picked.value.reduce((n, f) => n + (f.duration || 0), 0)))
const sizeKbps = computed(() => FORMATS[format.value].estimate || kbps.value)
const estimate = computed(() => seconds.value * sizeKbps.value * 1000 / 8)
const perHour = computed(() => 3600 * sizeKbps.value * 1000 / 8)

// ------------------------------------------------------------ queueing
const busy = ref(false)
const enqueued = ref(null)          // { position, count }

const canQueue = computed(() => !busy.value
  && (!custom.value || !!customDir.value.trim())
  && (mode.value === 'single' ? !!info.value : picked.value.length > 0))

async function enqueue() {
  busy.value = true
  try {
    const common = {
      options: { format: format.value, bitrate_kbps: kbps.value },
      output_mode: outputMode.value,
      output_dir: custom.value ? customDir.value.trim() : '',
    }
    const resp = mode.value === 'single'
      ? await api.enqueueAudio({ source: info.value.path, track: track.value, ...common })
      : await api.enqueueAudioBatch({
        path: scan.value.path, files: picked.value.map((f) => f.path),
        language: language.value, ...common,
      })
    enqueued.value = { position: resp.position, count: resp.count }
    ElMessage.success(resp.count > 1 ? `已加入列队：${resp.count} 个文件` : `已加入列队（第 ${resp.position} 位）`)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    busy.value = false
  }
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
  if (bytes >= 1 << 20) return (bytes / (1 << 20)).toFixed(bytes >= 100 << 20 ? 0 : 1) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

function fmtDuration(s) {
  s = Math.round(s || 0)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

function trackLine(t) {
  return [`#${t.index}`, t.language_name, t.title ? `「${t.title}」` : '', t.codec, t.channel_name]
    .filter(Boolean).join(' ')
}

function tracksShort(f) {
  return f.tracks.map((t) => t.language_name + (t.title ? `「${t.title}」` : '')).join('、')
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
        <el-input v-model="path" placeholder="要提取音频的视频文件的完整路径">
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
      <el-form-item label="格式">
        <el-radio-group v-model="format">
          <el-radio value="flac">无损 FLAC（推荐）</el-radio>
          <el-radio value="opus">Opus（体积小）</el-radio>
        </el-radio-group>
        <el-select v-if="FORMATS[format].rates.length" v-model="kbps" class="rate">
          <el-option
            v-for="r in FORMATS[format].rates" :key="r" :value="r"
            :label="`${r} kbps${r === FORMATS[format].default ? '（推荐）' : ''}`"
          />
        </el-select>
        <div class="hint below">
          一律 16 kHz 单声道（语音识别本来就先把音频转成这样再听），约 {{ fmtBytes(perHour) }}/小时。
          <template v-if="format === 'flac'">
            无损：解出来和直接翻译视频时识别听到的音频<strong>逐采样相同</strong>，翻这个文件就等于翻原视频。
          </template>
          <template v-else>
            体积约为 FLAC 的六分之一，有损：实测本地语音检测认出的对白与原视频有 2–3% 的出入。只在要传到别处、体积要紧时选它。
          </template>
        </div>
      </el-form-item>
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
            <code>Film.mkv</code> 旁边写出 <code>Film{{ FORMATS[format].suffix }}</code>。原文件不动；之后在「翻译任务」批量翻译这个文件夹时，同名的视频与音频只翻一次（用视频）。
          </template>
          <template v-else>
            保持原来的文件名；批量时每个文件在指定文件夹里保留原来的子文件夹。
          </template>
          已有同名文件时另起名（<code>Film.2{{ FORMATS[format].suffix }}</code>），绝不覆盖。
        </div>
      </el-form-item>
      <el-form-item v-if="mode === 'batch'">
        <el-button type="primary" :loading="scanning" :disabled="!folder.trim()" @click="scanFolder">
          列出文件夹里的视频
        </el-button>
      </el-form-item>
      <div class="hint below intro">
        把视频里的<strong>一条音轨</strong>提取成一个单独的文件，给翻译用：比整部片小得多，传到别处快，也能直接在「翻译任务」里当片源（纯音频片源）。
        只读音轨、不碰画面，一部电影通常一两分钟；和压制一样走列队，前面有任务时要排队等。
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
        <span class="file-meta">{{ fmtDuration(info.duration) }} · {{ fmtBytes(info.size) }}</span>
        <el-tag v-if="info.extracted" size="small" type="warning">已经提取过</el-tag>
      </div>
      <div v-if="info.extracted" class="file-meta extracted">
        旁边已有本程序从它提取的 <code>{{ info.extracted }}</code>；同样的格式和音轨再加入列队，会认出这个文件、不再重写。
      </div>
      <div class="tracks">
        <div class="tracks-title">提取哪条音轨</div>
        <el-radio-group v-model="track" class="track-list">
          <el-radio v-for="t in info.tracks" :key="t.index" :value="t.index">
            🔊 {{ trackLine(t) }}
            <el-tag v-if="t.default" size="small" type="info" class="tag">默认</el-tag>
          </el-radio>
        </el-radio-group>
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
      :title="`文件夹里还有 ${scan.discs.length} 张原盘（没有列进来）：请先到「原盘」页封装成 MKV`"
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
          <el-tag v-if="row.extracted" size="small" type="warning" class="tag">已提取</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="音轨" min-width="160">
        <template #default="{ row }">{{ tracksShort(row) }}</template>
      </el-table-column>
      <el-table-column label="时长" width="84">
        <template #default="{ row }">{{ fmtDuration(row.duration) }}</template>
      </el-table-column>
      <el-table-column label="大小" width="90" align="right">
        <template #default="{ row }">{{ fmtBytes(row.size) }}</template>
      </el-table-column>
    </el-table>
    <div v-if="scan.skipped.length" class="skipped">
      <div class="hint">以下 {{ scan.skipped.length }} 个读不出音轨，没有列进来：</div>
      <div v-for="s in scan.skipped" :key="s.path" class="reason">✕ <code>{{ s.path }}</code>：{{ s.reason }}</div>
    </div>
    <el-form v-if="files.length" label-width="110px" class="lang-form" @submit.prevent>
      <el-form-item label="音轨">
        <!-- '' reads as "nothing chosen" to el-select, so the default is the
             placeholder and clearing goes back to it -->
        <el-select v-model="language" class="lang" placeholder="每个文件的默认音轨" clearable>
          <el-option
            v-for="l in languages" :key="l.code" :value="l.code"
            :label="`${l.name}（${l.count} 个文件有）`"
          />
        </el-select>
        <div class="hint below">
          各文件的音轨编号不一样，批量时按语言选。
          <template v-if="withoutLanguage">
            已选的文件里有 {{ withoutLanguage }} 个没有这种语言的音轨，它们用各自的默认音轨，任务日志里会写明。
          </template>
        </div>
      </el-form-item>
    </el-form>
  </el-card>

  <el-card shadow="never" class="block">
    <div class="footer">
      <div class="summary">
        <template v-if="mode === 'single'">
          <template v-if="info">
            → <code>{{ outName }}</code>
            <template v-if="outFolder">，写到 <code>{{ outFolder }}</code></template>
            ，约 {{ fmtBytes(estimate) }}
          </template>
          <template v-else>先选一个视频</template>
        </template>
        <template v-else>
          已选 <strong>{{ picked.length }}</strong> 个文件，共 {{ fmtDuration(seconds) }}，提取后约
          <strong>{{ fmtBytes(estimate) }}</strong>
        </template>
      </div>
      <div class="buttons">
        <el-button type="primary" :disabled="!canQueue" :loading="busy" @click="enqueue()">
          ＋ 加入列队{{ mode === 'batch' && picked.length ? `（${picked.length} 个）` : '' }}
        </el-button>
      </div>
    </div>
    <el-alert v-if="enqueued" type="success" :closable="false" style="margin-top: 10px">
      <template #title>
        <template v-if="enqueued.count > 1">
          已加入列队：{{ enqueued.count }} 个文件，每个一条任务（从第 {{ enqueued.position }} 位起）。
        </template>
        <template v-else>已加入列队（第 {{ enqueued.position }} 位）。</template>
        <el-link type="primary" :underline="false" @click="emit('goto', 'queue')">去列队查看进度 →</el-link>
      </template>
    </el-alert>
  </el-card>

  <FileBrowser ref="fileBrowser" v-model="browserVisible" :mode="browseMode" @pick="onPick" />
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
.rate {
  width: 160px;
  margin-left: 16px;
}
.lang {
  width: 260px;
}
.lang-form {
  margin-top: 14px;
}
.custom-dir {
  margin-top: 6px;
  max-width: 560px;
}
.block {
  margin-top: 16px;
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
.extracted {
  margin-top: 6px;
}
.tracks {
  margin-top: 10px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.tracks-title {
  margin-bottom: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.track-list {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
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

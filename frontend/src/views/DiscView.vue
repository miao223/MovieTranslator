<script setup>
// 原盘：把蓝光 / DVD / ISO 重新封装成 MKV。
//
// Two ways in: one disc, or a folder (批量) whose discs are all listed. Either
// way the page only asks the server what the discs hold (/api/disc/scan,
// /api/disc/batch-scan — metadata only, so they are simply asked again
// whenever a switch here changes) and adds the chosen titles to the queue.
// Every decision about a disc — what is the film, what is a duplicate, what
// the files will be called, which volumes of a box set number on from each
// other — is the server's; this page shows it and lets the ticks be changed.
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import FileBrowser from '../components/FileBrowser.vue'
import DiscTitles from '../components/DiscTitles.vue'
import SubtitleDialog from '../components/SubtitleDialog.vue'

const emit = defineEmits(['goto'])

const mode = ref('single')        // 'single' | 'batch'

// ------------------------------------------------ where the MKVs go (both)
const outputMode = ref('beside')  // the settings' default, read on mount
const customDir = ref('')

onMounted(async () => {
  try {
    const s = await api.getSettings()
    outputMode.value = s.disc?.output_mode || 'beside'
    customDir.value = s.disc?.output_dir || ''
  } catch {
    /* the defaults above */
  }
})

function outputParams() {
  const custom = outputMode.value === 'custom'
  return { output_mode: outputMode.value, output_dir: custom ? customDir.value.trim() : '' }
}

// ------------------------------------------------------------ one disc
const path = ref('')
const report = ref(null)
const scanning = ref(false)
const scanError = ref('')
const name = ref('')              // '' = the disc's own, or its box set's
const series = ref(null)          // null = decided from the disc
const episodeStart = ref(null)    // null = automatic (1, or on from the volumes before)
const enqueued = ref(null)        // { position, count } after 加入列队
const busy = ref(false)

// ticks: the analysis's defaults, overridden title by title by the user.
// Kept by id, so a re-scan (new name, 整片/分集) keeps what was changed.
const picked = reactive({})
const manual = reactive({})

async function scan({ fresh = false } = {}) {
  const target = path.value.trim()
  if (!target) return
  if (fresh) {
    name.value = ''
    series.value = null
    episodeStart.value = null
    for (const k of Object.keys(manual)) delete manual[k]
  }
  scanning.value = true
  scanError.value = ''
  enqueued.value = null
  try {
    const out = outputParams()
    const r = await api.discScan({
      path: target, series: series.value, episodeStart: episodeStart.value,
      name: name.value, outputMode: out.output_mode, outputDir: out.output_dir,
    })
    report.value = r
    for (const k of Object.keys(picked)) delete picked[k]
    for (const t of r.titles) {
      picked[t.id] = t.selectable && (t.id in manual ? manual[t.id] : t.selected)
    }
  } catch (e) {
    report.value = null
    scanError.value = e.message
  } finally {
    scanning.value = false
  }
}

// a new disc: start from its own defaults
watch(path, () => {
  report.value = null
  scanError.value = ''
})

let nameTimer = null
watch(name, () => {
  if (!report.value) return
  clearTimeout(nameTimer)
  nameTimer = setTimeout(() => scan(), 400)
})
watch([series, episodeStart], () => { if (report.value) scan() })

function toggle(t, value) {
  manual[t.id] = value
  picked[t.id] = value
}

const chosen = computed(() => (report.value?.titles || []).filter((t) => picked[t.id]))
const chosenBytes = computed(() => chosen.value.reduce((n, t) => n + (t.size || 0), 0))
const sizeUnknown = computed(() => chosen.value.some((t) => t.size == null))
const tooBig = computed(() => {
  const r = report.value
  return !!r && r.free_bytes != null && chosenBytes.value + (1 << 30) > r.free_bytes
})
const canQueue = computed(() => {
  const r = report.value
  return !!r && !r.analysis_only && !r.encrypted && !!r.output_dir
    && chosen.value.length > 0 && !busy.value
})

// subtitles: the 「加入列队并做字幕」 dialog's answer, null for plain 加入列队
async function enqueue(subtitles = null) {
  const r = report.value
  if (!r) return
  busy.value = true
  try {
    const resp = await api.enqueueDisc({
      path: r.path,
      titles: chosen.value.map((t) => t.id),
      name: name.value,
      ...outputParams(),
      episode_start: episodeStart.value,
      series: series.value,
      subtitles,
    })
    enqueued.value = { position: resp.position, count: chosen.value.length, subtitles: !!subtitles }
    subDialog.value = false
    ElMessage.success(`已加入列队（第 ${resp.position} 位）`)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    busy.value = false
  }
}

// ------------------------------------------------------- a whole folder
const batchPath = ref('')
const recursive = ref(true)
const batch = ref(null)
const batchScanning = ref(false)
const batchError = ref('')
const batchQueued = ref(null)     // { count, position } after 加入列队
const answers = reactive({})      // disc path → { series, name, episode_start }
const include = reactive({})      // disc path → wanted at all
const bPicked = reactive({})      // disc path → { title id → ticked }
const bManual = reactive({})      // disc path → { title id → the user's own tick }
const unfolded = reactive({})     // disc path → card open

function clear(obj) {
  for (const k of Object.keys(obj)) delete obj[k]
}

function queueable(d) {
  return !d.encrypted && !d.analysis_only && d.titles.some((t) => t.selectable)
}

async function batchScan({ fresh = false } = {}) {
  const target = batchPath.value.trim()
  if (!target) return
  if (fresh) [answers, include, bPicked, bManual, unfolded].forEach(clear)
  batchScanning.value = true
  batchError.value = ''
  batchQueued.value = null
  try {
    const r = await api.discBatchScan({
      path: target, recursive: recursive.value, ...outputParams(),
      discs: Object.entries(answers).map(([p, a]) => ({ path: p, ...a })),
    })
    batch.value = r
    for (const d of r.discs) {
      answers[d.path] ??= { series: null, name: '', episode_start: null }
      if (!(d.path in include)) include[d.path] = queueable(d) && d.titles.some((t) => t.selected)
      const mine = (bManual[d.path] ??= {})
      bPicked[d.path] = Object.fromEntries(d.titles.map((t) => [
        t.id, t.selectable && (t.id in mine ? mine[t.id] : t.selected)]))
    }
    if (r.discs.length === 1) unfolded[r.discs[0].path] = true
  } catch (e) {
    batch.value = null
    batchError.value = e.message
  } finally {
    batchScanning.value = false
  }
}

watch(batchPath, () => {
  batch.value = null
  batchError.value = ''
})
watch(recursive, () => { if (batch.value) batchScan() })

let batchTimer = null
function rescanSoon(delay = 0) {
  clearTimeout(batchTimer)
  batchTimer = setTimeout(() => batchScan(), delay)
}

// A box set's name is the set's: typed on one volume, it is every volume's.
function setName(d, value) {
  const set = d.volume?.chained ? d.volume.id : null
  for (const x of batch.value.discs) {
    if (x === d || (set && x.volume?.chained && x.volume.id === set)) answers[x.path].name = value
  }
  rescanSoon(400)
}

function setAnswer(d, key, value) {
  answers[d.path][key] = value
  rescanSoon()
}

function bToggle(d, t, value) {
  bManual[d.path][t.id] = value
  bPicked[d.path][t.id] = value
}

function chosenOf(d) {
  return d.titles.filter((t) => bPicked[d.path]?.[t.id])
}

// "第 7–13 集" when the ticked episodes run on, "5 集" when they do not
function episodes(nums) {
  const lo = Math.min(...nums)
  const hi = Math.max(...nums)
  if (lo === hi) return `第 ${lo} 集`
  return hi - lo + 1 === nums.length ? `第 ${lo}–${hi} 集` : `${nums.length} 集`
}

// "第 7–13 集 + 花絮 2 个 · 约 36.4 GB"
function pickedLine(d) {
  const ts = chosenOf(d)
  if (!ts.length) return '没有勾选标题'
  const of = (category) => ts.filter((t) => t.category === category)
  const parts = []
  if (of('main').length) parts.push('正片')
  if (of('episode').length) parts.push(episodes(of('episode').map((t) => t.ordinal)))
  if (of('extra').length) parts.push(`花絮 ${of('extra').length} 个`)
  if (of('variant').length) parts.push(`另一版本 ${of('variant').length} 个`)
  const rest = ts.filter((t) => !['main', 'episode', 'extra', 'variant'].includes(t.category))
  if (rest.length) parts.push(`其他 ${rest.length} 个`)
  return parts.join(' + ') + ` · 约 ${fmtBytes(ts.reduce((n, t) => n + (t.size || 0), 0))}`
}

function discName(d) {
  const parts = d.root.split(/[\\/]/).filter(Boolean)
  return parts[parts.length - 1] || d.root
}

function whyNot(d) {
  if (d.encrypted) return '仍是加密状态，本程序不做解密，无法封装'
  if (d.analysis_only) return '只有元数据（没有视频文件），只能分析，不能封装'
  return '没有可以封装的标题'
}

const wanted = computed(() =>
  (batch.value?.discs || []).filter((d) => include[d.path] && chosenOf(d).length))
const wantedTitles = computed(() => wanted.value.reduce((n, d) => n + chosenOf(d).length, 0))
const wantedBytes = computed(() => wanted.value.reduce(
  (n, d) => n + chosenOf(d).reduce((m, t) => m + (t.size || 0), 0), 0))
// discs writing to the same folder share its free space
const batchTooBig = computed(() => {
  const need = {}
  const free = {}
  for (const d of wanted.value) {
    need[d.output_dir] = (need[d.output_dir] || 0)
      + chosenOf(d).reduce((m, t) => m + (t.size || 0), 0)
    free[d.output_dir] = d.free_bytes
  }
  return Object.keys(need).some((k) => free[k] != null && need[k] + (1 << 30) > free[k])
})
const canQueueBatch = computed(() =>
  wanted.value.length > 0 && !busy.value && wanted.value.every((d) => d.output_dir))

async function enqueueBatch(subtitles = null) {
  const b = batch.value
  if (!b) return
  busy.value = true
  try {
    const discs = b.discs.map((d) => {
      const titles = chosenOf(d).map((t) => t.id)
      return { path: d.path, ...answers[d.path], titles,
               include: !!include[d.path] && titles.length > 0 }
    })
    const resp = await api.enqueueDiscBatch({
      path: b.path, recursive: recursive.value, ...outputParams(), discs, subtitles,
    })
    batchQueued.value = { count: resp.count, position: resp.position, subtitles: !!subtitles }
    subDialog.value = false
    ElMessage.success(`已加入列队：${resp.count} 张盘`)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    busy.value = false
  }
}

// ------------------------------------------ 加入列队并做字幕 (both modes)
const subDialog = ref(false)

// what the dialog is about: the ticked titles of this disc, or of every
// wanted disc of the batch — they all get subtitles, extras included
const subTitles = computed(() => (mode.value === 'single'
  ? chosen.value
  : wanted.value.flatMap((d) => chosenOf(d))))

function languagesOf(kind) {
  const seen = new Set()
  for (const t of subTitles.value) {
    for (const s of t.streams) {
      if (s.kind === kind && s.carried) seen.add(s.language_name || '未标注')
    }
  }
  return [...seen]
}

function confirmSubtitles(options) {
  if (mode.value === 'single') enqueue(options)
  else enqueueBatch(options)
}

// ------------------------------------------------ both: output, browsing
let outputTimer = null
watch([outputMode, customDir], () => {
  clearTimeout(outputTimer)
  outputTimer = setTimeout(() => {
    if (mode.value === 'single' && report.value) scan()
    if (mode.value === 'batch' && batch.value) batchScan()
  }, 400)
})

const browserVisible = ref(false)
const browseMode = ref('disc')
const browseFor = ref('path')     // 'path' | 'batch' | 'output'
const fileBrowser = ref(null)

function openBrowser(target) {
  browseFor.value = target
  browseMode.value = { path: 'disc', batch: 'discs', output: 'dir' }[target]
  const start = target === 'output' ? (customDir.value || report.value?.output_dir || '')
    : target === 'batch' ? batchPath.value : ''
  setTimeout(() => fileBrowser.value?.open(start), 0)
}

function onPick(p) {
  if (browseFor.value === 'output') customDir.value = p
  else if (browseFor.value === 'batch') batchPath.value = p
  else path.value = p
}

const KIND = { bd: '蓝光', dvd: 'DVD' }

function fmtBytes(bytes) {
  if (bytes == null) return '—'
  if (bytes >= 1 << 30) return (bytes / (1 << 30)).toFixed(1) + ' GB'
  if (bytes >= 1 << 20) return (bytes / (1 << 20)).toFixed(0) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}
</script>

<template>
  <el-card shadow="never">
    <el-form label-width="110px" @submit.prevent>
      <el-form-item label="方式">
        <el-radio-group v-model="mode">
          <el-radio value="single">单张光盘</el-radio>
          <el-radio value="batch">批量（一个文件夹里的所有原盘）</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item v-if="mode === 'single'" label="原盘" required>
        <el-input
          v-model="path"
          placeholder="BDMV / VIDEO_TS 所在的文件夹，或 .iso 镜像的完整路径"
          @keyup.enter="scan({ fresh: true })"
        >
          <template #append>
            <el-button @click="openBrowser('path')">浏览…</el-button>
          </template>
        </el-input>
      </el-form-item>
      <template v-else>
        <el-form-item label="文件夹" required>
          <el-input
            v-model="batchPath"
            placeholder="放着原盘的文件夹：里面的蓝光 / DVD 文件夹和 .iso 会全部列出来"
            @keyup.enter="batchScan({ fresh: true })"
          >
            <template #append>
              <el-button @click="openBrowser('batch')">浏览…</el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item label="包含子文件夹">
          <el-switch v-model="recursive" />
        </el-form-item>
      </template>
      <el-form-item label="输出位置">
        <el-radio-group v-model="outputMode">
          <el-radio value="beside">放在光盘旁边</el-radio>
          <el-radio value="inside">放进各自的文件夹</el-radio>
          <el-radio value="custom">全部放到指定文件夹</el-radio>
        </el-radio-group>
        <el-input
          v-if="outputMode === 'custom'" v-model="customDir" class="custom-dir"
          placeholder="输出文件夹的完整路径"
        >
          <template #append>
            <el-button @click="openBrowser('output')">浏览…</el-button>
          </template>
        </el-input>
        <div class="hint below">
          <template v-if="outputMode === 'beside'">
            放在光盘所在的文件夹：<code>…/Film (1992)/</code> 旁边写出 <code>…/Film (1992).mkv</code>，
            <code>Film.iso</code> 旁边写出 <code>Film.mkv</code>；多卷合集的各卷会放在一起。
          </template>
          <template v-else-if="outputMode === 'inside'">
            放进光盘自己的文件夹（BDMV / VIDEO_TS 所在的那一层）；.iso 镜像会在旁边建一个同名文件夹。
          </template>
          <template v-else>所有光盘的 MKV 都写进这一个文件夹。</template>
          设置页可以改默认的输出位置。
        </div>
      </el-form-item>
      <el-form-item>
        <el-button
          v-if="mode === 'single'" type="primary" :loading="scanning" :disabled="!path.trim()"
          @click="scan({ fresh: true })"
        >分析这张盘</el-button>
        <el-button
          v-else type="primary" :loading="batchScanning" :disabled="!batchPath.trim()"
          @click="batchScan({ fresh: true })"
        >分析文件夹里的原盘</el-button>
      </el-form-item>
      <div class="hint below intro">
        把蓝光 / DVD 原盘<strong>重新封装成 MKV</strong>：音视频原样拷贝、不重编码，章节、音轨和字幕的语言一并保留。
        <strong>不做任何解密</strong>，仍是加密状态的原盘会被识别出来并拒绝。
      </div>
    </el-form>
    <el-alert v-if="mode === 'single' && scanError" type="error" :closable="false" :title="scanError" />
    <el-alert v-if="mode === 'batch' && batchError" type="error" :closable="false" :title="batchError" />
  </el-card>

  <!-- ============================================================ one disc -->
  <el-card v-if="mode === 'single' && report" shadow="never" class="report-card">
    <div class="disc-head">
      <span class="disc-name">💿 {{ KIND[report.kind] }}{{ report.source === 'iso' ? '镜像' : '' }}</span>
      <code class="disc-path">{{ report.root }}</code>
      <el-tag v-if="report.label" size="small" type="info">盘上的标题：{{ report.label }}</el-tag>
      <el-tag v-if="report.volume?.chained" size="small" type="warning">
        多卷合集 {{ report.volume.index }}/{{ report.volume.count }}
      </el-tag>
    </div>
    <p class="mode">
      <el-tag size="small" :type="report.mode === 'series' ? 'warning' : 'success'">
        {{ report.mode === 'series' ? '剧集' : '电影' }}
      </el-tag>
      {{ report.mode_reason }}
    </p>
    <p v-if="report.volume" class="mode">{{ report.volume.note }}</p>
    <el-alert
      v-if="report.encrypted" type="error" :closable="false" show-icon style="margin-bottom: 10px"
      title="这张盘仍是加密状态，本程序不做解密，无法封装"
    />
    <el-alert
      v-for="w in report.warnings.filter((w) => !w.includes('加密'))" :key="w"
      type="warning" :closable="false" show-icon :title="w" style="margin-bottom: 10px"
    />

    <el-form label-width="110px" class="options" @submit.prevent>
      <el-form-item label="片名">
        <el-input v-model="name" :placeholder="report.name" style="max-width: 420px" />
        <span class="hint">输出文件名的开头，默认用光盘文件夹的名字（多卷合集用各卷名字的共同部分）</span>
        <!-- a ripped folder is often named after the volume label
             (SHOW_S1_D1); the disc's own title reads better -->
        <el-link
          v-if="report.label && report.label !== report.name" type="primary" :underline="false"
          style="margin-left: 8px" @click="name = report.label"
        >用盘上的标题「{{ report.label }}」</el-link>
      </el-form-item>
      <el-form-item v-if="report.series_choice" label="整片 / 分集">
        <el-radio-group :model-value="series === null ? report.mode === 'series' : series"
                        @update:model-value="series = $event">
          <el-radio :value="false">整片导出</el-radio>
          <el-radio :value="true">分集导出</el-radio>
        </el-radio-group>
        <span class="hint">
          自动判断是「{{ report.series_default ? '分集' : '整片' }}」<template v-if="series !== null">，
          <el-link type="primary" :underline="false" @click="series = null">恢复自动</el-link></template>
        </span>
      </el-form-item>
      <el-form-item v-if="report.mode === 'series'" label="起始集号">
        <el-input-number
          :model-value="episodeStart ?? report.episode_start" :min="0" :max="9999" size="small"
          @update:model-value="episodeStart = $event ?? null"
        />
        <span class="hint">
          <template v-if="episodeStart === null">
            自动{{ report.volume?.chained && report.episode_start > 1 ? '：接着同一套的前几卷编号' : '' }}
          </template>
          <template v-else>
            你指定的，<el-link type="primary" :underline="false" @click="episodeStart = null">恢复自动</el-link>
          </template>
        </span>
      </el-form-item>
    </el-form>

    <DiscTitles :titles="report.titles" :picked="picked" @toggle="toggle" />

    <div class="footer">
      <div class="summary">
        已选 <strong>{{ chosen.length }}</strong> 个，约 <strong>{{ fmtBytes(chosenBytes) }}</strong>
        <template v-if="sizeUnknown">（部分大小未知）</template>
        <template v-if="report.output_dir">
          → <code>{{ report.output_dir }}</code>
          <template v-if="report.free_bytes != null">（剩余 {{ fmtBytes(report.free_bytes) }}）</template>
        </template>
        <el-tag v-if="tooBig" type="danger" size="small" style="margin-left: 6px">空间可能不够</el-tag>
      </div>
      <div class="buttons">
        <el-button type="primary" :disabled="!canQueue" :loading="busy" @click="enqueue()">
          ＋ 加入列队
        </el-button>
        <el-button type="primary" plain :disabled="!canQueue" @click="subDialog = true">
          ＋ 加入列队并做字幕
        </el-button>
      </div>
    </div>
    <el-alert
      v-if="report.analysis_only" type="info" :closable="false" style="margin-top: 10px"
      title="这是只有元数据的副本（没有视频文件）：可以看分析结果，但不能封装"
    />
    <el-alert v-if="enqueued" type="success" :closable="false" style="margin-top: 10px">
      <template #title>
        已加入列队（第 {{ enqueued.position }} 位），{{ enqueued.count }} 个标题。
        <template v-if="enqueued.subtitles">封装完成后，这些 MKV 会自动加入字幕任务（排在列队最后）。</template>
        <el-link type="primary" :underline="false" @click="emit('goto', 'queue')">去列队查看进度 →</el-link>
      </template>
    </el-alert>
  </el-card>

  <!-- ======================================================= a whole folder -->
  <el-card v-if="mode === 'batch' && batch" shadow="never" class="report-card">
    <div class="disc-head">
      <span class="disc-name">📁 找到 {{ batch.discs.length }} 张原盘</span>
      <code class="disc-path">{{ batch.path }}</code>
    </div>

    <div v-for="d in batch.discs" :key="d.path" class="disc-card">
      <div class="disc-row">
        <el-checkbox
          :model-value="!!include[d.path]" :disabled="!queueable(d)"
          @update:model-value="include[d.path] = $event"
        />
        <span class="disc-title" @click="unfolded[d.path] = !unfolded[d.path]">
          💿 {{ discName(d) }}
        </span>
        <el-tag size="small" type="info">{{ KIND[d.kind] }}{{ d.source === 'iso' ? '镜像' : '' }}</el-tag>
        <el-tag v-if="d.volume?.chained" size="small" type="warning">
          合集 {{ d.volume.index }}/{{ d.volume.count }}
        </el-tag>
        <el-tag size="small" :type="d.mode === 'series' ? 'warning' : 'success'">
          {{ d.mode === 'series' ? '剧集' : '电影' }}
        </el-tag>
        <span class="disc-sum">{{ queueable(d) ? pickedLine(d) : whyNot(d) }}</span>
        <el-link type="primary" :underline="false" class="unfold"
                 @click="unfolded[d.path] = !unfolded[d.path]">
          {{ unfolded[d.path] ? '收起 ▴' : '展开 ▾' }}
        </el-link>
      </div>
      <div class="disc-sub">
        {{ d.name }} → <code>{{ d.output_dir || '（还没有选输出文件夹）' }}</code>
        <template v-if="d.volume"> · {{ d.volume.note }}</template>
      </div>
      <div v-if="unfolded[d.path]" class="disc-body">
        <p class="mode">{{ d.mode_reason }}</p>
        <el-alert
          v-for="w in d.warnings.filter((w) => !w.includes('加密'))" :key="w"
          type="warning" :closable="false" show-icon :title="w" style="margin-bottom: 8px"
        />
        <el-form label-width="100px" class="options" size="small" @submit.prevent>
          <el-form-item label="片名">
            <el-input
              :model-value="answers[d.path].name" :placeholder="d.name" style="max-width: 360px"
              @update:model-value="setName(d, $event)"
            />
            <el-link
              v-if="d.label && d.label !== d.name" type="primary" :underline="false"
              style="margin-left: 8px" @click="setName(d, d.label)"
            >用盘上的标题「{{ d.label }}」</el-link>
          </el-form-item>
          <el-form-item v-if="d.series_choice" label="整片 / 分集">
            <el-radio-group
              :model-value="answers[d.path].series === null ? d.mode === 'series' : answers[d.path].series"
              @update:model-value="setAnswer(d, 'series', $event)"
            >
              <el-radio :value="false">整片导出</el-radio>
              <el-radio :value="true">分集导出</el-radio>
            </el-radio-group>
            <el-link
              v-if="answers[d.path].series !== null" type="primary" :underline="false"
              style="margin-left: 8px" @click="setAnswer(d, 'series', null)"
            >恢复自动</el-link>
          </el-form-item>
          <el-form-item v-if="d.mode === 'series'" label="起始集号">
            <el-input-number
              :model-value="answers[d.path].episode_start ?? d.episode_start" :min="0" :max="9999"
              @update:model-value="setAnswer(d, 'episode_start', $event ?? null)"
            />
            <el-link
              v-if="answers[d.path].episode_start !== null" type="primary" :underline="false"
              style="margin-left: 8px" @click="setAnswer(d, 'episode_start', null)"
            >恢复自动</el-link>
            <span v-else class="hint">自动</span>
          </el-form-item>
        </el-form>
        <DiscTitles
          :titles="d.titles" :picked="bPicked[d.path]"
          @toggle="(t, v) => bToggle(d, t, v)"
        />
      </div>
    </div>

    <div v-if="batch.skipped.length" class="skipped">
      <div class="hint">以下 {{ batch.skipped.length }} 个读不出来，没有列进来：</div>
      <div v-for="s in batch.skipped" :key="s.path" class="reason">
        ✕ <code>{{ s.path }}</code>：{{ s.reason }}
      </div>
    </div>

    <div class="footer">
      <div class="summary">
        已选 <strong>{{ wanted.length }}</strong> 张盘、<strong>{{ wantedTitles }}</strong> 个标题，
        约 <strong>{{ fmtBytes(wantedBytes) }}</strong>
        <el-tag v-if="batchTooBig" type="danger" size="small" style="margin-left: 6px">空间可能不够</el-tag>
      </div>
      <div class="buttons">
        <el-button type="primary" :disabled="!canQueueBatch" :loading="busy" @click="enqueueBatch()">
          ＋ 加入列队（{{ wanted.length }} 张盘）
        </el-button>
        <el-button type="primary" plain :disabled="!canQueueBatch" @click="subDialog = true">
          ＋ 加入列队并做字幕
        </el-button>
      </div>
    </div>
    <el-alert v-if="batchQueued" type="success" :closable="false" style="margin-top: 10px">
      <template #title>
        已加入列队：{{ batchQueued.count }} 张盘，每张一条任务（从第 {{ batchQueued.position }} 位起）。
        <template v-if="batchQueued.subtitles">每张盘封装完成后，它的 MKV 会自动加入字幕任务（排在列队最后）。</template>
        <el-link type="primary" :underline="false" @click="emit('goto', 'queue')">去列队查看进度 →</el-link>
      </template>
    </el-alert>
  </el-card>

  <FileBrowser ref="fileBrowser" v-model="browserVisible" :mode="browseMode" @pick="onPick" />
  <SubtitleDialog
    v-model="subDialog" :busy="busy" :count="subTitles.length"
    :extras="subTitles.filter((t) => t.category === 'extra').length"
    :discs="mode === 'single' ? 1 : wanted.length"
    :audio="languagesOf('audio')" :subtitles="languagesOf('subtitle')"
    @confirm="confirmSubtitles"
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
}
.hint.intro {
  margin: 0 0 4px;
}
.custom-dir {
  margin-top: 6px;
  max-width: 560px;
}
.report-card {
  margin-top: 16px;
}
.disc-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.disc-name {
  font-weight: 600;
  font-size: 15px;
}
.disc-path {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  word-break: break-all;
}
.mode {
  color: var(--el-text-color-regular);
  font-size: 13px;
}
.options {
  margin-top: 6px;
}
.disc-card {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--el-border-color-light);
  border-radius: var(--app-radius);
}
.disc-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.disc-title {
  font-weight: 600;
  cursor: pointer;
  word-break: break-all;
}
.disc-sum {
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.unfold {
  margin-left: auto;
}
.disc-sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  word-break: break-all;
}
.disc-body {
  margin-top: 8px;
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
  margin-top: 16px;
  flex-wrap: wrap;
}
.summary {
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
</style>

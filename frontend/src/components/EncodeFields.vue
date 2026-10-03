<script setup>
// 压制参数: what a re-encode does, bound to an EncodeOptions object
// (schemas.py) through v-model. One form for three places — the 压制 page,
// the settings page's defaults and the 原盘 page's full-pipeline dialog.
//
// Which encoders exist, the quality scale each one counts in and whether it
// does 10-bit all come from the server (encode.capabilities): a GPU encoder
// is offered only on a machine that can really open it, and x265 is never
// offered the "film" tuning it refuses.
import { computed, onMounted, ref } from 'vue'
import { AUDIO_LANGS, SUB_LANGS } from '../languages'
import { encoderCaps, FAMILIES, VENDORS, familyOf } from '../encode'

const opts = defineModel({ type: Object, required: true })
const props = defineProps({
  // the full pipeline writes MKV only: MP4 cannot carry a disc's picture
  // subtitles, which the subtitle step after it reads
  lockContainer: { type: Boolean, default: false },
  // one probed file (GET /api/encode/probe), for the hints that depend on it
  source: { type: Object, default: null },
  // what 按画面选 means where this form is shown (settings / one file / batch)
  autoHint: { type: String, default: '' },
  // on the 修复 page, which has its own deinterlace and size fields
  // (RestoreFields): these two are not shown here twice
  restore: { type: Boolean, default: false },
})

const encoders = ref([])
const deinterlaceOk = ref(true)
const loadError = ref('')

onMounted(async () => {
  try {
    const caps = await encoderCaps()
    encoders.value = caps.encoders
    deinterlaceOk.value = caps.deinterlace
  } catch (e) {
    loadError.value = e.message
  }
})

const current = computed(() => encoders.value.find((e) => e.id === opts.value.video_codec))
const family = computed(() => familyOf(encoders.value, opts.value.video_codec))
const families = computed(() =>
  FAMILIES.filter((f) => encoders.value.some((e) => e.family === f.value)))
const choices = computed(() => encoders.value.filter((e) => e.family === family.value))
// a default saved on another machine can name an encoder this one lacks
const missing = computed(() => encoders.value.length && opts.value.video_codec !== 'copy'
  && !current.value)
const reencoding = computed(() => opts.value.video_codec !== 'copy')
const scale = computed(() => current.value?.quality || { min: 0, max: 51, default: 22, name: 'CRF' })
const tunes = computed(() => current.value?.tunes || [])
const hardware = computed(() => !!current.value?.hardware)

function engineLabel(e) {
  return e.hardware ? `显卡：${VENDORS[e.id.split('_').pop()] || e.engine}` : `CPU：${e.engine}`
}

function chooseFamily(value) {
  if (value === 'copy') {
    opts.value.video_codec = 'copy'
    return
  }
  const first = encoders.value.find((e) => e.family === value)
  if (first) chooseEncoder(first.id)
}

// a new encoder counts quality on its own scale: start from its own
// recommendation rather than carry a number that means something else
function chooseEncoder(id) {
  opts.value.video_codec = id
  const e = encoders.value.find((x) => x.id === id)
  if (!e) return
  opts.value.quality = e.quality.default
  if (!e.tunes.includes(opts.value.tune)) opts.value.tune = ''
  if (opts.value.bit_depth === '10' && !e.ten_bit) opts.value.bit_depth = 'auto'
}

const PRESETS = [
  { value: 'ultrafast', label: '最快' },
  { value: 'fast', label: '较快' },
  { value: 'medium', label: '中等（推荐）' },
  { value: 'slow', label: '较慢' },
  { value: 'veryslow', label: '最慢' },
]
const TUNES = { film: '电影（真人）', animation: '动画', grain: '保留胶片颗粒' }
const HEIGHTS = [
  { value: 0, label: '保持原分辨率' },
  { value: 2160, label: '不超过 2160p（4K）' },
  { value: 1440, label: '不超过 1440p' },
  { value: 1080, label: '不超过 1080p' },
  { value: 720, label: '不超过 720p' },
  { value: 576, label: '不超过 576p' },
  { value: 480, label: '不超过 480p' },
]
const AUDIO = [
  { value: 'copy', label: '原样保留' },
  { value: 'eac3', label: 'E-AC-3（兼容性最好的环绕声）' },
  { value: 'aac', label: 'AAC' },
  { value: 'libopus', label: 'Opus（同样体积音质最好）' },
  { value: 'ac3', label: 'AC-3' },
  { value: 'flac', label: 'FLAC（无损）' },
]
const AUTO_KBPS = {
  eac3: '立体声 224k、5.1 640k（7.1 降为 5.1）',
  ac3: '立体声 224k、5.1 640k（7.1 降为 5.1）',
  aac: '立体声 192k、5.1 384k、7.1 512k',
  libopus: '立体声 128k、5.1 256k、7.1 320k',
}
// the language pickers name tracks, so the "each file's default" entry of
// the translation lists has no place here
const AUDIO_PICK = AUDIO_LANGS.filter((l) => l.value)
const SUB_PICK = SUB_LANGS.filter((l) => l.value)

const depthAuto = computed(() =>
  ['H.265', 'AV1'].includes(family.value) ? '10bit' : '8bit')

// what the chosen file itself says about these choices
const video = computed(() => props.source?.video || null)
const lossless = computed(() => (props.source?.audio || []).filter((a) => a.lossless))
const ntscInterlaced = computed(() => video.value && video.value.interlaced >= 0.5
  && Math.abs(video.value.fps - 29.97) < 0.05)
</script>

<template>
  <el-form label-width="96px" class="encode-fields" @submit.prevent>
    <el-alert v-if="loadError" type="error" :closable="false" style="margin-bottom: 10px"
              :title="`读不到本机可用的编码器：${loadError}`" />
    <el-form-item v-if="!lockContainer" label="容器">
      <el-radio-group v-model="opts.container">
        <el-radio value="mkv">MKV（推荐）</el-radio>
        <el-radio value="mp4">MP4（兼容性最好）</el-radio>
      </el-radio-group>
      <div v-if="opts.container === 'mp4'" class="hint block warn">
        MP4 装不下字幕轨和字体附件，片源里的这些会被略去；需要保留请用 MKV。
      </div>
    </el-form-item>

    <el-form-item label="按画面选">
      <el-switch v-model="opts.auto_pick" />
      <span class="hint">让视觉模型看一张截图拼图，判断是动画还是真人、颗粒轻重，按实测选编码、速度和质量</span>
      <div v-if="opts.auto_pick && autoHint" class="hint block">{{ autoHint }}</div>
    </el-form-item>

    <el-form-item label="视频编码">
      <el-radio-group :model-value="family" @update:model-value="chooseFamily">
        <el-radio v-for="f in families" :key="f.value" :value="f.value">{{ f.label }}</el-radio>
        <el-radio value="copy">保持原样（不重编码）</el-radio>
      </el-radio-group>
      <div v-if="!encoders.length && !loadError" class="hint block">正在读取本机可用的编码器…</div>
    </el-form-item>
    <el-form-item v-if="reencoding && (choices.length || missing)" label="编码器">
      <el-select
        :model-value="current ? opts.video_codec : ''" style="width: 260px"
        :placeholder="missing ? `${opts.video_codec}（本机不可用）` : ''"
        @update:model-value="chooseEncoder"
      >
        <el-option v-for="e in choices" :key="e.id" :value="e.id" :label="engineLabel(e)" />
      </el-select>
      <span v-if="missing" class="hint warn">{{ opts.video_codec }} 在本机不可用，请另选一个</span>
      <span v-else-if="hardware" class="hint">显卡编码快得多，同样体积下画质略逊于 CPU</span>
      <span v-else-if="choices.length === 1" class="hint">本机没有可用的显卡编码器，用 CPU 编码</span>
    </el-form-item>

    <template v-if="reencoding">
      <el-form-item label="画质">
        <el-radio-group v-model="opts.rate_control">
          <el-radio value="quality">恒定质量（推荐）</el-radio>
          <el-radio value="bitrate">平均码率</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item v-if="opts.rate_control === 'quality'" :label="scale.name">
        <el-slider
          v-model="opts.quality" :min="scale.min" :max="scale.max" :step="1" show-input
          style="width: 420px"
        />
        <span class="hint">越小越清晰、文件越大；这个编码器推荐 {{ scale.default }}</span>
      </el-form-item>
      <el-form-item v-else label="码率">
        <el-input-number v-model="opts.bitrate_kbps" :min="100" :max="200000" :step="500" />
        <span class="hint">kbps（只算画面）。1080p 常见 4000–8000，4K 常见 12000–25000</span>
      </el-form-item>
      <el-form-item label="速度">
        <el-select v-model="opts.preset" style="width: 160px">
          <el-option v-for="p in PRESETS" :key="p.value" :value="p.value" :label="p.label" />
        </el-select>
        <span class="hint">越慢，同样画质下文件越小</span>
      </el-form-item>
    </template>

    <el-form-item label="音频">
      <el-select v-model="opts.audio_codec" style="width: 260px">
        <el-option v-for="a in AUDIO" :key="a.value" :value="a.value" :label="a.label" />
      </el-select>
    </el-form-item>
    <template v-if="opts.audio_codec !== 'copy'">
      <el-form-item label="转哪些音轨">
        <el-radio-group v-model="opts.audio_scope">
          <el-radio value="lossless">只转无损音轨（推荐）</el-radio>
          <el-radio value="all">全部音轨</el-radio>
        </el-radio-group>
        <div class="hint block">
          无损音轨是 TrueHD、DTS-HD MA、FLAC、PCM 这类，一条常有几 GB；AC-3、DTS 这类有损音轨再转一遍只会更差，
          「只转无损」时原样保留。已经是 {{ AUDIO.find((a) => a.value === opts.audio_codec)?.label.split('（')[0] }}
          的音轨不会再转。
        </div>
      </el-form-item>
      <el-form-item v-if="opts.audio_codec !== 'flac'" label="音频码率">
        <el-input-number v-model="opts.audio_bitrate_kbps" :min="0" :max="6144" :step="32" />
        <span class="hint">
          kbps，每条音轨；0 = 自动（{{ AUTO_KBPS[opts.audio_codec] }}）
        </span>
      </el-form-item>
      <el-form-item label="声道">
        <el-radio-group v-model="opts.audio_mixdown">
          <el-radio value="keep">保持</el-radio>
          <el-radio value="stereo">降为立体声</el-radio>
        </el-radio-group>
        <span v-if="opts.audio_mixdown === 'stereo'" class="hint">多声道的音轨都会转码并降为立体声</span>
      </el-form-item>
    </template>

    <el-collapse class="more">
      <el-collapse-item
        name="more"
        :title="restore ? '更多选项：位深、内容类型、保留哪些音轨和字幕'
          : '更多选项：位深、分辨率、反交错、内容类型、保留哪些音轨和字幕'"
      >
        <template v-if="reencoding">
          <el-form-item label="位深">
            <el-radio-group v-model="opts.bit_depth">
              <el-radio value="auto">自动（{{ depthAuto }}）</el-radio>
              <el-radio value="8">8bit</el-radio>
              <el-radio value="10" :disabled="!!current && !current.ten_bit">10bit</el-radio>
            </el-radio-group>
            <span v-if="current && !current.ten_bit" class="hint">这个编码器在本机不支持 10bit</span>
            <span v-else class="hint">
              H.265 / AV1 用 10bit 能减少色带；H.264 10bit 很多设备放不了。HDR 片源总是 10bit
            </span>
          </el-form-item>
          <el-form-item v-if="!restore" label="分辨率">
            <el-select v-model="opts.max_height" style="width: 200px">
              <el-option v-for="h in HEIGHTS" :key="h.value" :value="h.value" :label="h.label" />
            </el-select>
            <span class="hint">只缩小、不放大，画面比例不变</span>
          </el-form-item>
          <el-form-item v-if="!restore" label="反交错">
            <el-select v-model="opts.deinterlace" style="width: 340px" :disabled="!deinterlaceOk">
              <el-option value="auto" label="自动：只处理标记为隔行的帧（推荐）" />
              <el-option value="off" label="关闭" />
              <el-option value="all" label="每一帧都反交错" />
              <el-option value="ivtc" label="反胶片过带（IVTC）" />
              <el-option value="bob" label="还原 60 帧（仅隔行录像，见「修复」页）" />
              <el-option value="match" label="只做场匹配（30 帧逐行、两场错开）" />
              <el-option value="detect" label="按画面自动判断（开压前分析，约 20 秒）" />
            </el-select>
            <div class="hint block">
              <template v-if="opts.deinterlace === 'bob'">
                一个场还原成一帧：隔行的 VHS、电视录像变成每秒 59.94（PAL 50）帧。胶片拍的电影不要用这个。
              </template>
              <template v-else-if="opts.deinterlace === 'match'">
                30 帧逐行拍摄、存进 DVD 时两个场错开了一场的片子：重新配对还原成 29.97 帧逐行，不删帧也不加倍。
              </template>
              <template v-else-if="opts.deinterlace === 'detect'">
                开压前看画面判断是胶片过带、30 帧错场、真隔行还是逐行，各用各的方式；判断结果写在任务日志里。
              </template>
              <template v-else-if="opts.deinterlace === 'ivtc'">
                把 NTSC DVD 上胶片转制的电影、动画还原成每秒 23.976 帧（5 帧里去掉 1 帧）。
                只对帧率 29.97、时间戳均匀的片源生效，其余的自动改用普通反交错。
              </template>
              <template v-else>逐行的片源（蓝光的 1080p 电影、网络片源）不受影响；DVD 和 1080i 的隔行画面会被还原成逐行。</template>
            </div>
          </el-form-item>
          <el-form-item v-if="tunes.length" label="内容类型">
            <!-- '' is a real choice; el-select shows the placeholder for it -->
            <el-select v-model="opts.tune" placeholder="默认" style="width: 200px">
              <el-option value="" label="默认" />
              <el-option v-for="t in tunes" :key="t" :value="t" :label="TUNES[t] || t" />
            </el-select>
          </el-form-item>
        </template>
        <el-form-item label="保留音轨">
          <el-select
            v-model="opts.audio_languages" multiple clearable style="width: 420px"
            placeholder="全部保留"
          >
            <el-option v-for="l in AUDIO_PICK" :key="l.value" :value="l.value" :label="l.label" />
          </el-select>
          <div class="hint block">不选就全部保留；没有语言标记的音轨总会保留，筛完一条不剩时也全部保留</div>
        </el-form-item>
        <el-form-item label="保留字幕轨">
          <el-radio-group v-model="opts.subtitles">
            <el-radio value="all">全部</el-radio>
            <el-radio value="languages">只保留这些语言</el-radio>
            <el-radio value="none">都不要</el-radio>
          </el-radio-group>
          <el-select
            v-if="opts.subtitles === 'languages'" v-model="opts.subtitle_languages" multiple
            style="width: 420px; margin-top: 6px" placeholder="选择要保留的字幕语言"
          >
            <el-option v-for="l in SUB_PICK" :key="l.value" :value="l.value" :label="l.label" />
          </el-select>
          <div class="hint block">章节、字体附件和封面图总会原样保留（MP4 装不下的除外）</div>
        </el-form-item>
      </el-collapse-item>
    </el-collapse>

    <template v-if="source">
      <el-alert
        v-if="video && video.hdr" type="warning" :closable="false" show-icon class="source-hint"
        :title="`片源是 ${video.hdr}`"
      >
        重编码会保持 10bit 和 HDR 色彩标记；x265 / SVT-AV1 会带上 HDR10 的母版元数据，显卡编码器带不上。
        杜比视界层和 HDR10+ 动态元数据会丢失；只有杜比视界 profile 5（没有 HDR10 基础层）的片子无法压制。
        想原封不动请选「保持原样」。
      </el-alert>
      <el-alert
        v-if="ntscInterlaced && !restore && !['ivtc', 'bob', 'match', 'detect'].includes(opts.deinterlace)"
        type="info" :closable="false"
        show-icon class="source-hint" title="片源是隔行的 NTSC（29.97）"
      >
        如果这是电影或动画（胶片转制的 DVD），「更多选项 → 反交错」选「反胶片过带」画质更好。
      </el-alert>
      <el-alert
        v-if="lossless.length && opts.audio_codec === 'copy'" type="info" :closable="false"
        show-icon class="source-hint"
        :title="`片源有 ${lossless.length} 条无损音轨（${lossless.map((a) => a.profile || a.codec).join('、')}），原样保留会占不少空间`"
      />
    </template>
  </el-form>
</template>

<style scoped>
.hint {
  margin-left: 12px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.hint.block {
  display: block;
  width: 100%;
  margin: 4px 0 0;
  line-height: 1.6;
}
.warn {
  color: var(--el-color-warning);
}
.more {
  margin: 4px 0 12px;
}
.more :deep(.el-collapse-item__header) {
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.source-hint {
  margin-top: 8px;
}
</style>

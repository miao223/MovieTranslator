<script setup>
// 修复参数: the picture steps a restore adds before the encode — what kind
// of source it is (which decides the deinterlacing), field order, crop,
// pixel shape, denoise, enlarging. Bound to the same EncodeOptions object as
// EncodeFields (schemas.py); the server does all of it (services/restore.py).
import { computed } from 'vue'
import { effectiveDeinterlace } from '../encode'

const opts = defineModel({ type: Object, required: true })
const props = defineProps({
  // one probed file (GET /api/encode/probe), for the resulting size
  source: { type: Object, default: null },
  // whether the 修复 engine is set up (settings.restore.engine_python)
  engineReady: { type: Boolean, default: false },
  // what its picture is (GET /api/restore/analyze); null while measuring
  analysis: { type: Object, default: null },
  analyzing: { type: Boolean, default: false },
})

const MODE_NAMES = {
  bob: '还原 60 帧', ivtc: '反胶片过带', match: '只做场匹配', off: '不反交错', auto: '自动反交错',
}

const KINDS = [
  {
    value: 'detect',
    label: '自动判断（推荐）',
    hint: '开压前先看画面：是胶片过带、30 帧错场、真隔行还是逐行，每张碟各用各的方式。'
      + 'DVD 不管里面是什么都标成「隔行」，所以不看标记，看画面。批量处理一整个文件夹时只有这样每张都对。',
  },
  {
    value: 'bob',
    label: '摄像机拍的录像：还原 60 帧',
    hint: 'VHS、电视节目、家庭录像。隔行的录像每秒本来就有 60 个（PAL 是 50 个）真实的场，'
      + '一场还原成一帧，动作就是当年拍下来的那样流畅——不是 AI 补出来的帧。',
  },
  {
    value: 'ivtc',
    label: '胶片拍的电影：反胶片过带，还原 24 帧',
    hint: 'DVD 上的电影正片多数是这种：胶片每秒 24 格，为了上 29.97 的电视被重复了一些场。'
      + '去掉重复、还原成 23.976 帧；对它用「还原 60 帧」只会让画面一顿一顿的。只对 29.97 的片源生效。',
  },
  {
    value: 'match',
    label: '30 帧逐行、两场错开：只做场匹配',
    hint: '2000 年代以后很多电视剧用 30 帧逐行拍摄，存进 DVD 时两个场错开了一场，每一帧看起来都是梳齿，'
      + '其实一帧都没丢。把场重新配对就还原成 29.97 帧逐行，不删帧也不加倍；配不上的片段照常反交错。',
  },
  {
    value: 'auto',
    label: '已经是逐行的，或拿不准',
    hint: '只处理标记为隔行的帧，帧率不变。',
  },
]
const kindHint = computed(() => KINDS.find((k) => k.value === opts.value.deinterlace)?.hint
  || '反交错方式在「压制参数」里另选过；这里选一种就会覆盖它。')

const MODELS = [
  {
    value: 'lanczos', label: 'Lanczos（不用 AI）',
    hint: '传统插值，和播放器自己放大的效果差不多；不会编造任何细节。实测 VHS 采集、VHS 翻录和混场片源上，'
      + 'AI 的闪烁是它的 1.5–3.4 倍，这几种片源用它最稳。',
  },
  {
    value: 'realviformer', label: 'RealViformer（视频模型）',
    hint: '看之前的帧一起修。实测 DVD 上画面最自然、闪烁在三个 AI 里最低（5 段里 4 段），'
      + '但会改招牌、手写字的笔画。RTX 5070 Ti 约 2.6 帧/秒（fp32）。MIT 许可。',
  },
  {
    value: 'realbasicvsr', label: 'RealBasicVSR（视频模型）',
    hint: '前后的帧都看，按窗口处理。清理压缩方块和噪点最狠，但皮肤发蜡、纹理像油画。'
      + 'RTX 5070 Ti 约 2.3–3 帧/秒。Apache 许可。',
  },
  {
    value: 'liveaction_span', label: '2xLiveActionV1_SPAN（单帧，DVD 专用）',
    hint: '专门修 DVD 的压缩方块、光晕、色度，改动最小、最快（RTX 5070 Ti 约 23 帧/秒，'
      + '实际多半由 CPU 决定）；噪点重的片源上闪烁比另外两个高。CC-BY-NC-SA：仅限非商业使用。',
  },
]
// ai_model "" is Lanczos; el-select reads "" as nothing chosen and shows its
// placeholder, so the select works on a name for it
const model = computed({
  get: () => opts.value.ai_model || 'lanczos',
  set: (v) => { opts.value.ai_model = v === 'lanczos' ? '' : v },
})
const modelHint = computed(() => MODELS.find((m) => m.value === model.value)?.hint || '')

const UPSCALE = [
  { value: 0, label: '不放大' },
  { value: 720, label: '720p' },
  { value: 1080, label: '1080p' },
  { value: 1440, label: '1440p' },
  { value: 2160, label: '2160p（4K）' },
]

const cadence = computed(() => props.analysis?.cadence || null)
const borders = computed(() => props.analysis?.borders || [0, 0, 0, 0])
const anyBorder = computed(() => borders.value.some((b) => b > 0))

// what the server will make of this file (encode.geometry, restore.name_tags)
// — shown only, never sent
function even(v) {
  return Math.max(2, Math.round(v / 2) * 2)
}
function gcd(a, b) {
  return b ? gcd(b, a % b) : a
}
const result = computed(() => {
  const v = props.source?.video
  if (!v) return ''
  const o = opts.value
  const auto = o.crop_auto && props.analysis ? borders.value : [0, 0, 0, 0]
  const top = o.crop_top + auto[0]
  const bottom = o.crop_bottom + auto[1]
  const left = o.crop_left + auto[2]
  const right = o.crop_right + auto[3]
  const w = v.width - left - right
  const h = v.height - top - bottom
  if (w < 16 || h < 16) return '裁边太多'
  let num = 1
  let den = 1
  if (o.aspect === '4:3' || o.aspect === '16:9') {
    const [a, b] = o.aspect === '4:3' ? [4, 3] : [16, 9]
    num = a * v.height
    den = b * v.width
  } else if (v.sar) {
    [num, den] = v.sar.split(':').map(Number)
    if (!num || !den) [num, den] = [1, 1]
  }
  const g = gcd(num, den) || 1
  let size = `${w}×${h}${num === den ? '' : `（像素比例 ${num / g}:${den / g}）`}`
  const shown = (w * num) / den
  const short = Math.min(shown, h)
  if (o.upscale && short < o.upscale) {
    const f = o.upscale / short
    size = `${even(shown * f)}×${even(h * f)}`
  }
  const mode = effectiveDeinterlace(o, props.analysis)
  let fps = v.fps
  if (mode === 'bob' && fps && fps <= 31) fps *= 2
  else if (mode === 'ivtc' && Math.abs(fps - 29.97) < 0.05) fps = 23.976
  const note = o.deinterlace === 'detect' && !props.analysis ? '（帧率等分析完才知道）' : ''
  return `${size} · ${Math.round(fps * 1000) / 1000} fps${note}`
})
</script>

<template>
  <el-form label-width="96px" class="restore-fields" @submit.prevent>
    <el-form-item label="片源">
      <el-radio-group v-model="opts.deinterlace" class="kinds">
        <el-radio v-for="k in KINDS" :key="k.value" :value="k.value">{{ k.label }}</el-radio>
      </el-radio-group>
      <div class="hint block">{{ kindHint }}</div>
      <div v-if="analyzing" class="measured">🔍 正在分析这个文件的画面（约 20 秒）…</div>
      <div v-else-if="cadence" class="measured">
        测得：<strong>{{ cadence.name }}</strong>
        <template v-if="cadence.mode">→ {{ MODE_NAMES[cadence.mode] }}</template>
        <span class="hint">（梳齿帧 {{ Math.round(cadence.combed * 100) }}%，场匹配后 {{ Math.round(cadence.matched * 100) }}%）</span>
      </div>
      <el-alert
        v-if="cadence?.kind === 'blended'" type="warning" :closable="false" show-icon class="note"
        title="这张碟疑似「混场转制」：胶片转成录像带时，相邻的画面被混进了同一个场，找不回原来的帧"
      >
        画面里本来就带着淡淡的重影，任何反交错都去不掉；AI 放大会把重影和画面一起锐化。
        这类片子建议不用 AI，只做反交错 + 编码。
      </el-alert>
    </el-form-item>
    <el-form-item v-if="['bob', 'detect'].includes(opts.deinterlace)" label="场序">
      <el-select v-model="opts.field_order" style="width: 240px">
        <el-option value="auto" label="自动：从画面测（推荐）" />
        <el-option value="tff" label="上场优先" />
        <el-option value="bff" label="下场优先" />
      </el-select>
      <div class="hint block">
        两个场谁先拍的。采集卡录下来的文件常常不标或标错，所以默认从画面上量（日志里写着测得的结果）。
        成品里的动作如果一前一后地抖，就是场序反了，改选另一种重压。
      </div>
    </el-form-item>
    <el-form-item label="黑边">
      <el-switch v-model="opts.crop_auto" />
      <span class="hint">开压前自动找出上下左右的黑边并裁掉</span>
      <div v-if="opts.crop_auto && analysis" class="measured">
        <template v-if="anyBorder">
          测得黑边：上 {{ borders[0] }} 下 {{ borders[1] }} 左 {{ borders[2] }} 右 {{ borders[3] }}
        </template>
        <template v-else>没有测到黑边</template>
      </div>
    </el-form-item>
    <el-form-item :label="opts.crop_auto ? '另外再裁' : '裁边'">
      <div class="crop">
        <span>上</span><el-input-number v-model="opts.crop_top" :min="0" :max="1000" :step="2" step-strictly size="small" />
        <span>下</span><el-input-number v-model="opts.crop_bottom" :min="0" :max="1000" :step="2" step-strictly size="small" />
        <span>左</span><el-input-number v-model="opts.crop_left" :min="0" :max="1000" :step="2" step-strictly size="small" />
        <span>右</span><el-input-number v-model="opts.crop_right" :min="0" :max="1000" :step="2" step-strictly size="small" />
      </div>
      <div class="hint block">
        像素，按片源原来的尺寸算，只能是双数。VHS 画面最下面那几行跳动的杂波（磁头切换）不是黑色、测不出来，
        一般手动裁 8–12。不裁的话，放大和降噪会把杂波当成画面一起「修」。
      </div>
    </el-form-item>
    <el-form-item label="画面比例">
      <el-select v-model="opts.aspect" style="width: 220px">
        <el-option value="auto" label="按文件里的标记" />
        <el-option value="4:3" label="4:3" />
        <el-option value="16:9" label="16:9" />
      </el-select>
      <div v-if="analysis?.aspect_missing" class="hint block warn">
        这个文件没有标记像素比例（{{ analysis.width }}×{{ analysis.height }} 会被当成方形像素，画面被拉扁或压窄）。
        采集或转码时丢了标记，请选它本来的样子：普通电视画面是 4:3，宽银幕是 16:9（指整个画面，含黑边）。
      </div>
      <div v-else class="hint block">只在文件的比例标记错了或丢了时才改。指的是整个画面（含黑边）的比例。</div>
    </el-form-item>
    <el-form-item label="降噪">
      <el-radio-group v-model="opts.denoise">
        <el-radio value="off">不降噪</el-radio>
        <el-radio value="light">轻度（时域）</el-radio>
        <el-radio value="strong">强力（频域 + 时域）</el-radio>
      </el-radio-group>
      <div class="hint block">
        轻度只抹掉帧与帧之间跳动的噪点，几乎不伤细节；强力去得更多，画面也更软。DVD 一般不用开；
        两档的参数还没拿真实录像带实测过。
      </div>
    </el-form-item>
    <el-form-item label="放大到">
      <el-select v-model="opts.upscale" style="width: 160px">
        <el-option v-for="u in UPSCALE" :key="u.value" :value="u.value" :label="u.label" />
      </el-select>
      <div class="hint block">按短边算，并换成方形像素（720×480 宽银幕 → 1920×1080）。</div>
    </el-form-item>
    <el-form-item v-if="opts.upscale" label="放大方式">
      <el-select v-model="model" style="width: 300px">
        <el-option
          v-for="m in MODELS" :key="m.value" :value="m.value" :label="m.label"
          :disabled="m.value !== 'lanczos' && !engineReady"
        />
      </el-select>
      <div class="hint block">{{ modelHint }}</div>
      <div v-if="!engineReady" class="hint block warn">
        AI 模型要先在「设置 → 修复引擎」里配置引擎；现在只能用 Lanczos。
      </div>
      <div v-else class="hint block">
        三个模型各有长短，换一部片结果就不一样：拿不准就用下面的「试看」在这部片上比一比再选。
        AI 放大一部片要十几到几十个小时。
      </div>
    </el-form-item>
    <el-form-item v-if="result" label="成品">
      <span class="result">{{ result }}</span>
    </el-form-item>
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
.kinds {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}
.measured {
  width: 100%;
  margin-top: 4px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.note {
  margin-top: 8px;
}
.crop {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.crop .el-input-number {
  width: 110px;
}
.result {
  font-size: 13px;
  color: var(--el-text-color-regular);
}
</style>

<script setup>
// 原盘页「加入列队并压制、做字幕」: the whole way from a disc to a film with
// subtitles, through the queue. Each disc is remuxed to lossless MKV, its
// MKVs are encoded right behind it (so one disc's lossless files take up
// space at a time), and each encode then queues its subtitles at the end.
//
// The encode starts from the settings page's defaults (AppSettings.encode)
// and is MKV only here: the subtitle step may read the disc's picture
// subtitles, which MP4 cannot carry.
import { onMounted, reactive, ref } from 'vue'
import { api } from '../api'
import { defaultEncodeOptions } from '../encode'
import EncodeFields from './EncodeFields.vue'
import SubtitleFields, { newSubtitleOptions } from './SubtitleFields.vue'

defineProps({
  modelValue: { type: Boolean, default: false },
  count: { type: Number, default: 0 },          // MKVs the remux will write
  extras: { type: Number, default: 0 },
  discs: { type: Number, default: 1 },
  audio: { type: Array, default: () => [] },
  subtitles: { type: Array, default: () => [] },
  busy: { type: Boolean, default: false },
})
const emit = defineEmits(['update:modelValue', 'confirm'])

const encodeOpts = ref({ ...defaultEncodeOptions() })
const keepLossless = ref(false)       // the user's default: the lossless MKV goes
const subOpts = reactive(newSubtitleOptions(true))

onMounted(async () => {
  try {
    const s = await api.getSettings()
    if (s.encode) encodeOpts.value = { ...s.encode, container: 'mkv' }
  } catch {
    /* the built-in defaults */
  }
})

function confirm() {
  emit('confirm', {
    encode: { options: { ...encodeOpts.value, container: 'mkv' }, keep_lossless: keepLossless.value },
    subtitles: { ...subOpts },
  })
}
</script>

<template>
  <el-dialog
    :model-value="modelValue" title="封装、压制、做字幕" width="760px" top="4vh"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <p class="lead">
      {{ discs > 1 ? `这 ${discs} 张盘` : '这张盘' }}会封装出 <strong>{{ count }}</strong> 个 MKV<template
        v-if="extras">（其中花絮 {{ extras }} 个）</template>，每个都在列队里依次：
      <strong>① 封装成无损 MKV → ② 紧接着压制 → ③ 做字幕</strong>。
      压制任务排在它那张盘后面，字幕任务排在列队最后。
    </p>
    <p v-if="audio.length || subtitles.length" class="hint">
      <template v-if="audio.length">盘上的音轨：{{ audio.join('、') }}</template>
      <template v-if="audio.length && subtitles.length">；</template>
      <template v-if="subtitles.length">字幕：{{ subtitles.join('、') }}</template>
    </p>

    <div class="section">
      <div class="section-title">压制</div>
      <EncodeFields
        v-model="encodeOpts" lock-container
        auto-hint="每个 MKV 开压前各让视觉模型看一次画面（每个一次调用）；判断不了时用下面这些参数。"
      />
      <div class="keep">
        <el-switch :model-value="!keepLossless" @update:model-value="keepLossless = !$event" />
        <span class="keep-label">压制成功后删除无损 MKV，压制版沿用原来的文件名</span>
        <div class="hint block">
          <template v-if="!keepLossless">
            只在压制版通过校验（帧数、时长、轨道、章节都对得上）之后才替换；校验不过就两份都留下，并在列队里说明。
            光盘本身不动，无损版随时能重新封装出来。
          </template>
          <template v-else>
            无损 MKV 留着，压制版另起名（<code>片名.E01.HEVC.mkv</code>），字幕跟着压制版命名。
          </template>
        </div>
      </div>
      <div class="hint block">默认参数在「设置 → 视频压制」里改。</div>
    </div>

    <div class="section">
      <div class="section-title">做字幕</div>
      <SubtitleFields v-model="subOpts" series-scope="disc" />
    </div>

    <template #footer>
      <el-button @click="emit('update:modelValue', false)">取消</el-button>
      <el-button
        type="primary" :loading="busy" :disabled="!subOpts.target_language.trim()"
        @click="confirm"
      >＋ 加入列队</el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.lead {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--el-text-color-regular);
  line-height: 1.7;
}
.hint {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.hint.block {
  display: block;
  margin: 4px 0 0;
  line-height: 1.6;
}
.section {
  margin-top: 14px;
  padding: 12px 14px 4px;
  border: 1px solid var(--el-border-color-light);
  border-radius: var(--app-radius);
}
.section-title {
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--el-text-color-primary);
}
.keep {
  margin: 0 0 12px 96px;
}
.keep-label {
  margin-left: 8px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
</style>

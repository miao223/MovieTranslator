<script setup>
// 「加入列队并做字幕」's dialog: what the files of a remux (原盘) or an
// encode (压制) are translated with once they are written. The same options
// as a translation batch on the 翻译任务 page, starting from the same
// defaults — the app saves no defaults of its own for these. The choices
// are kept while the page stays open; a reload starts from the defaults
// again, like the 翻译任务 form.
import { reactive } from 'vue'
import SubtitleFields, { newSubtitleOptions } from './SubtitleFields.vue'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  count: { type: Number, default: 0 },          // files that will get subtitles
  extras: { type: Number, default: 0 },         // …of which extras
  discs: { type: Number, default: 1 },
  audio: { type: Array, default: () => [] },    // languages the titles carry, for the hint
  subtitles: { type: Array, default: () => [] },
  busy: { type: Boolean, default: false },
  after: { type: String, default: '封装' },     // '封装' (原盘) or '压制'
  // whose table of names the files share (SubtitleFields)
  seriesScope: { type: String, default: 'disc' },
})
const emit = defineEmits(['update:modelValue', 'confirm'])

// one disc's episodes share names by default; a folder of files may be
// unrelated films, so a batch starts with sharing off (as on 翻译任务)
const opts = reactive(newSubtitleOptions(props.seriesScope === 'disc'))

function confirm() {
  emit('confirm', { ...opts })
}
</script>

<template>
  <el-dialog
    :model-value="modelValue" :title="`${after}完成后做字幕`" width="640px"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <p v-if="after === '封装'" class="lead">
      {{ discs > 1 ? `这 ${discs} 张盘` : '这张盘' }}封装出的 <strong>{{ count }}</strong> 个 MKV<template
        v-if="extras">（其中花絮 {{ extras }} 个）</template>，
      封装完成后会按下面的选项自动加入字幕任务，<strong>排在列队最后</strong>。
    </p>
    <p v-else class="lead">
      这 <strong>{{ count }}</strong> 个视频压制完成后，会按下面的选项自动加入字幕任务，<strong>排在列队最后</strong>。
    </p>
    <p v-if="audio.length || subtitles.length" class="hint">
      <template v-if="audio.length">音轨：{{ audio.join('、') }}</template>
      <template v-if="audio.length && subtitles.length">；</template>
      <template v-if="subtitles.length">字幕：{{ subtitles.join('、') }}</template>
    </p>
    <SubtitleFields v-model="opts" :series-scope="seriesScope" />
    <template #footer>
      <el-button @click="emit('update:modelValue', false)">取消</el-button>
      <el-button type="primary" :loading="busy" :disabled="!opts.target_language.trim()" @click="confirm">
        ＋ 加入列队
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.lead {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--el-text-color-regular);
  line-height: 1.6;
}
.hint {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>

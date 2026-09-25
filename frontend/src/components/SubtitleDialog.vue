<script setup>
// 「加入列队并做字幕」's dialog: what the MKVs of a remux are translated
// with once they are written. The same options as a translation batch on
// the 翻译任务 page, starting from the same defaults — the app saves no
// defaults of its own for these. The choices are kept while the page stays
// open; a reload starts from the defaults again, like the 翻译任务 form.
import { reactive } from 'vue'
import { AUDIO_LANGS, SOURCE_LANGS, SUB_LANGS, TARGET_LANGS } from '../languages'

defineProps({
  modelValue: { type: Boolean, default: false },
  count: { type: Number, default: 0 },          // MKVs that will get subtitles
  extras: { type: Number, default: 0 },         // …of which extras
  discs: { type: Number, default: 1 },
  audio: { type: Array, default: () => [] },    // languages the titles carry, for the hint
  subtitles: { type: Array, default: () => [] },
  busy: { type: Boolean, default: false },
})
const emit = defineEmits(['update:modelValue', 'confirm'])

const opts = reactive({
  text_source: 'asr',
  audio_language: '',
  subtitle_language: '',
  source_language: 'auto',
  target_language: '简体中文',
  output_mode: 'bilingual',
  embed_subtitle: false,
  series_mode: true,
})

function confirm() {
  emit('confirm', { ...opts })
}
</script>

<template>
  <el-dialog
    :model-value="modelValue" title="封装完成后做字幕" width="640px"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <p class="lead">
      {{ discs > 1 ? `这 ${discs} 张盘` : '这张盘' }}封装出的 <strong>{{ count }}</strong> 个 MKV<template
        v-if="extras">（其中花絮 {{ extras }} 个）</template>，
      封装完成后会按下面的选项自动加入字幕任务，<strong>排在列队最后</strong>。
    </p>
    <p v-if="audio.length || subtitles.length" class="hint">
      <template v-if="audio.length">盘上的音轨：{{ audio.join('、') }}</template>
      <template v-if="audio.length && subtitles.length">；</template>
      <template v-if="subtitles.length">字幕：{{ subtitles.join('、') }}</template>
    </p>
    <el-form label-width="96px" @submit.prevent>
      <el-form-item label="原文来源">
        <el-radio-group v-model="opts.text_source">
          <el-radio value="asr">语音识别</el-radio>
          <el-radio value="subtitle">片源已有的字幕</el-radio>
        </el-radio-group>
        <div v-if="opts.text_source === 'subtitle'" class="hint block">
          读取 MKV 里的字幕轨当原文（蓝光 / DVD 的字幕是图片，会先 OCR）；
          哪一集没有可用的字幕，就改用语音识别。
        </div>
      </el-form-item>
      <el-form-item label="音轨">
        <!-- '' (each file's default) is a real choice, but el-select shows
             its placeholder for it: make the placeholder say it -->
        <el-select v-model="opts.audio_language" :placeholder="AUDIO_LANGS[0].label" style="width: 260px">
          <el-option v-for="l in AUDIO_LANGS" :key="l.value" :value="l.value" :label="l.label" />
        </el-select>
      </el-form-item>
      <el-form-item v-if="opts.text_source === 'subtitle'" label="字幕轨">
        <el-select v-model="opts.subtitle_language" :placeholder="SUB_LANGS[0].label" style="width: 260px">
          <el-option v-for="l in SUB_LANGS" :key="l.value" :value="l.value" :label="l.label" />
        </el-select>
      </el-form-item>
      <el-form-item label="音频语言">
        <el-select v-model="opts.source_language" filterable allow-create style="width: 260px">
          <el-option v-for="l in SOURCE_LANGS" :key="l.value" :value="l.value" :label="l.label" />
        </el-select>
      </el-form-item>
      <el-form-item label="目标语言">
        <el-select v-model="opts.target_language" filterable allow-create style="width: 260px">
          <el-option v-for="l in TARGET_LANGS" :key="l" :value="l" :label="l" />
        </el-select>
      </el-form-item>
      <el-form-item label="字幕形式">
        <el-radio-group v-model="opts.output_mode">
          <el-radio value="bilingual">双语（原文 + 译文）</el-radio>
          <el-radio value="translation_only">纯译文</el-radio>
          <el-radio value="original_only">纯原文（不翻译）</el-radio>
          <el-radio value="bilingual_split">双语分离（两个文件）</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="输出形式">
        <el-radio-group v-model="opts.embed_subtitle">
          <el-radio :value="false">独立字幕文件</el-radio>
          <el-radio :value="true">合成带字幕的新视频</el-radio>
        </el-radio-group>
        <div v-if="opts.embed_subtitle" class="hint block warn">
          每个 MKV 旁边会<strong>再写一份同样大小</strong>的带字幕 MKV（不重编码），占用的空间翻倍。
        </div>
      </el-form-item>
      <el-form-item label="统一人名">
        <el-switch v-model="opts.series_mode" />
        <span class="hint">
          同一张盘（多卷合集是整套）共用一份人名 / 术语译名表，先译出的集数定下的译法后面沿用
        </span>
      </el-form-item>
    </el-form>
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
  margin-left: 12px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
p.hint {
  margin-left: 0;
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
</style>

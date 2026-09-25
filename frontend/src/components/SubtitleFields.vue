<script>
// What a translation queued after a remux or an encode starts from: the
// 翻译任务 form's own defaults — the app saves none for these, on purpose.
export function newSubtitleOptions(seriesMode = true) {
  return {
    text_source: 'asr',
    audio_language: '',
    subtitle_language: '',
    source_language: 'auto',
    target_language: '简体中文',
    output_mode: 'bilingual',
    embed_subtitle: false,
    series_mode: seriesMode,
  }
}
</script>

<script setup>
// The fields of a follow-up translation (schemas.DiscSubtitles), bound
// through v-model. The same options as a translation batch on the 翻译任务
// page; shared by SubtitleDialog (做字幕 after a remux or an encode) and
// PipelineDialog (the 原盘 page's remux → encode → subtitles).
import { AUDIO_LANGS, SOURCE_LANGS, SUB_LANGS, TARGET_LANGS } from '../languages'

const opts = defineModel({ type: Object, required: true })
defineProps({
  // whose table of names the files share: 'disc' — one disc, or a whole
  // box set; 'batch' — one 压制 batch; 'none' — a single file, nothing to share
  seriesScope: { type: String, default: 'disc' },
})
</script>

<template>
  <el-form label-width="96px" @submit.prevent>
    <el-form-item label="原文来源">
      <el-radio-group v-model="opts.text_source">
        <el-radio value="asr">语音识别</el-radio>
        <el-radio value="subtitle">片源已有的字幕</el-radio>
      </el-radio-group>
      <div v-if="opts.text_source === 'subtitle'" class="hint block">
        读取视频里的字幕轨当原文（蓝光 / DVD 的字幕是图片，会先 OCR）；
        哪个文件没有可用的字幕，就改用语音识别。
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
        每个视频旁边会<strong>再写一份同样大小</strong>的带字幕 MKV（不重编码），占用的空间翻倍。
      </div>
    </el-form-item>
    <el-form-item v-if="seriesScope !== 'none'" label="统一人名">
      <el-switch v-model="opts.series_mode" />
      <span v-if="seriesScope === 'disc'" class="hint">
        同一张盘（多卷合集是整套）共用一份人名 / 术语译名表，先译出的集数定下的译法后面沿用
      </span>
      <span v-else class="hint">
        这一批文件共用一份人名 / 术语译名表：适合整季剧集；文件夹里是互不相干的电影时请关掉
      </span>
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
</style>

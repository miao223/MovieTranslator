<script setup>
import { onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const settings = ref(null)
const preview = ref('')
const saving = ref(false)
let previewTimer = null

async function refreshPreview() {
  if (!settings.value) return
  try {
    const r = await api.promptPreview({
      prompts: settings.value.prompts,
      target_language: '简体中文',
      synopsis: '（这里会插入任务页填写的剧情简介）',
      max_line_chars: settings.value.subtitle.max_chars_per_line,
    })
    preview.value = r.prompt
  } catch (e) {
    preview.value = '预览失败: ' + e.message
  }
}

onMounted(async () => {
  try {
    settings.value = await api.getSettings()
    refreshPreview()
    watch(
      () => settings.value.prompts,
      () => {
        clearTimeout(previewTimer)
        previewTimer = setTimeout(refreshPreview, 400)
      },
      { deep: true }
    )
  } catch (e) {
    ElMessage.error('加载设置失败: ' + e.message)
  }
})

async function save() {
  saving.value = true
  try {
    settings.value = await api.saveSettings(settings.value)
    ElMessage.success('提示词设置已保存')
  } catch (e) {
    ElMessage.error('保存失败: ' + e.message)
  } finally {
    saving.value = false
  }
}

const DEFAULT_TONE = '语言口语化、符合角色语气，适合字幕阅读，简洁不啰嗦。'

function resetDefaults() {
  Object.assign(settings.value.prompts, {
    fix_asr_errors: true,
    link_fragments: true,
    normalize_loanwords: true,
    limit_length: true,
    tone: DEFAULT_TONE,
    glossary: '',
    extra: '',
    custom_system_prompt: '',
  })
  ElMessage.info('已恢复默认，记得点保存')
}
</script>

<template>
  <el-row v-if="settings" :gutter="16">
    <el-col :span="13">
      <el-card shadow="never" class="section">
        <template #header>🩹 语音识别缺陷补偿</template>
        <el-form label-width="140px">
          <el-form-item label="同音词纠错">
            <el-switch v-model="settings.prompts.fix_asr_errors" />
            <span class="hint">明显不通顺处按上下文推断本意，不硬译识别错误的词</span>
          </el-form-item>
          <el-form-item label="碎行衔接">
            <el-switch v-model="settings.prompts.link_fragments" />
            <span class="hint">相邻行常是半句话，要求跨行语序通顺、仍逐行输出</span>
          </el-form-item>
          <el-form-item label="外来语规范">
            <el-switch v-model="settings.prompts.normalize_loanwords" />
            <span class="hint">片假名等音译词全片统一译法（对日语尤其有效）</span>
          </el-form-item>
          <el-form-item label="译文长度限制">
            <el-switch v-model="settings.prompts.limit_length" />
            <span class="hint">按「设置→字幕→每行最大字符数」约束译文长度</span>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="section">
        <template #header>🎭 翻译风格</template>
        <el-input
          v-model="settings.prompts.tone"
          type="textarea"
          :rows="2"
          placeholder="例：大陆院线字幕风格，口语化，禁用网络流行语"
        />
      </el-card>

      <el-card shadow="never" class="section">
        <template #header>📖 自定义术语表（可选）</template>
        <el-input
          v-model="settings.prompts.glossary"
          type="textarea"
          :rows="4"
          placeholder="每行一条，格式：原文 → 译文&#10;例：Gandalf → 甘道夫&#10;ミサカ → 御坂"
        />
        <div class="hint" style="margin: 6px 0 0">
          模型仍会自动生成术语表；这里填的条目优先级更高，适合固定译名的系列作品
        </div>
      </el-card>

      <el-card shadow="never" class="section">
        <template #header>➕ 附加指令（可选）</template>
        <el-input
          v-model="settings.prompts.extra"
          type="textarea"
          :rows="3"
          placeholder="例：歌词保留原文并在后面附中文；粗口按语气弱化处理"
        />
      </el-card>

      <el-card shadow="never" class="section">
        <template #header>⚠️ 高级：完全自定义系统提示词</template>
        <el-input
          v-model="settings.prompts.custom_system_prompt"
          type="textarea"
          :rows="6"
          placeholder="留空则使用上方选项自动拼装。填写后将完全覆盖上方所有设置（开关与风格均失效）。&#10;可用占位符：{target_language} 目标语言、{synopsis} 剧情简介。&#10;注意：必须保留「每行输出 [行号] 译文」的格式要求，否则解析会失败。"
        />
      </el-card>

      <el-button type="primary" size="large" :loading="saving" @click="save">保存设置</el-button>
      <el-button size="large" @click="resetDefaults">恢复默认</el-button>
    </el-col>

    <el-col :span="11">
      <el-card shadow="never" class="preview-card">
        <template #header>👁️ 最终系统提示词预览（实时）</template>
        <pre class="preview">{{ preview }}</pre>
      </el-card>
    </el-col>
  </el-row>
  <el-skeleton v-else :rows="8" animated />
</template>

<style scoped>
.section {
  margin-bottom: 14px;
}
.hint {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
.preview-card {
  position: sticky;
  top: 12px;
}
.preview {
  white-space: pre-wrap;
  word-break: break-all;
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
  line-height: 1.7;
  color: #303133;
  background: #f5f7fa;
  border-radius: 6px;
  padding: 12px;
  margin: 0;
  max-height: 70vh;
  overflow-y: auto;
}
</style>

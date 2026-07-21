<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import { THEMES, applyTheme, loadTheme } from '../theme'

const theme = ref(loadTheme())

function pickTheme(value) {
  theme.value = applyTheme(value) // 立即生效并写入 localStorage，不随「保存设置」
}

const settings = ref(null)
const saving = ref(false)
const testing = ref(false)
const modelDownloaded = ref(null) // null = unknown / loading
const cuda = ref(null) // { available, device_count }
const storageInfo = ref(null) // { effective_dir, is_default }
const logInfo = ref(null) // { dir, files: [{name, size, modified}] }
const download = ref({ status: 'idle', progress: 0 })
let pollTimer = null

// size = download size in MB (from the HuggingFace repos, model.bin + config)
// VAD presets: [threshold, speech_pad_ms, min_speech_ms, min_silence_ms]
const VAD_PRESETS = [
  { name: '宽松·防漏（默认）', desc: '适合大多数电影：阈值 0.35 对偏小声的对白也能捕捉', v: [0.35, 400, 250, 2000] },
  { name: '标准', desc: 'faster-whisper 原始默认值（阈值 0.5），语音清晰、录音质量好的片源', v: [0.5, 400, 250, 2000] },
  { name: '极宽松·气声对白', desc: '悄悄话/气声仍被漏掉时用（阈值 0.25 + 加大填充）；嘈杂片源可能误检', v: [0.25, 800, 100, 1500] },
  { name: '严格·防噪', desc: '配乐音效嘈杂、出现幻听字幕时用（阈值 0.6）；小声对白可能被丢弃', v: [0.6, 300, 300, 2500] },
]

function applyVadPreset(p) {
  settings.value.asr.vad_threshold = p.v[0]
  settings.value.asr.vad_speech_pad_ms = p.v[1]
  settings.value.asr.vad_min_speech_ms = p.v[2]
  settings.value.asr.vad_min_silence_ms = p.v[3]
  ElMessage.info(`已应用预设「${p.name}」，记得保存`)
}

const WHISPER_MODELS = [
  { value: 'tiny', size: 75 },
  { value: 'base', size: 141 },
  { value: 'small', size: 464 },
  { value: 'medium', size: 1460 },
  { value: 'large-v2', size: 2946 },
  { value: 'large-v3', size: 2948 },
  { value: 'large-v3-turbo', size: 1547 },
  { value: 'distil-large-v3', size: 1446 },
  { value: 'CrisperWhisper', size: 2948 },
]

function fmtModelSize(mb) {
  return mb >= 1000 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB'
}
const COMPUTE_TYPES = ['int8', 'int8_float16', 'float16', 'float32']

// preview box is ~1/3 of a 1080p frame's height, scale fonts accordingly
const PREVIEW_SCALE = 0.33
const transStyle = computed(() => ({
  fontSize: Math.round(settings.value.subtitle.font_size * PREVIEW_SCALE) + 'px',
  color: settings.value.subtitle.translation_color,
  textShadow: '1px 1px 2px #000',
}))
const origStyle = computed(() => ({
  fontSize: Math.round(settings.value.subtitle.original_font_size * PREVIEW_SCALE) + 'px',
  color: settings.value.subtitle.original_color,
  textShadow: '1px 1px 2px #000',
}))

async function refreshModelStatus() {
  modelDownloaded.value = null
  try {
    const r = await api.modelStatus(settings.value.asr.model_size)
    modelDownloaded.value = r.downloaded
  } catch {
    modelDownloaded.value = null
  }
}

onMounted(async () => {
  try {
    settings.value = await api.getSettings()
    refreshModelStatus()
    api.cudaStatus().then((r) => (cuda.value = r)).catch(() => {})
    api.storageInfo().then((r) => (storageInfo.value = r)).catch(() => {})
    api.logs().then((r) => (logInfo.value = r)).catch(() => {})
  } catch (e) {
    ElMessage.error('加载设置失败: ' + e.message)
  }
})

async function save() {
  saving.value = true
  try {
    settings.value = await api.saveSettings(settings.value)
    ElMessage.success('设置已保存')
    api.storageInfo().then((r) => (storageInfo.value = r)).catch(() => {})
    refreshModelStatus()
  } catch (e) {
    ElMessage.error('保存失败: ' + e.message)
  } finally {
    saving.value = false
  }
}

async function startDownload() {
  try {
    download.value = await api.downloadModel(settings.value.asr.model_size)
    pollDownload()
  } catch (e) {
    ElMessage.error('启动下载失败: ' + e.message)
  }
}

function pollDownload() {
  clearInterval(pollTimer)
  let lastBytes = -1
  let stallCount = 0
  pollTimer = setInterval(async () => {
    try {
      download.value = await api.downloadStatus(settings.value.asr.model_size)
      const bytes = download.value.downloaded_bytes || 0
      if (download.value.status === 'downloading') {
        if (bytes === lastBytes) {
          stallCount += 1
          if (stallCount === 60) {
            ElMessage.warning('下载超过 1 分钟无进展，可能无法直连 HuggingFace——建议启用「模型下载走代理」后重试，或使用本地模型目录')
          }
        } else {
          lastBytes = bytes
          stallCount = 0
        }
      }
      if (download.value.status === 'done') {
        clearInterval(pollTimer)
        ElMessage.success('模型下载完成')
        refreshModelStatus()
      } else if (download.value.status === 'failed') {
        clearInterval(pollTimer)
        ElMessage.error('下载失败: ' + (download.value.error || '未知错误'))
      }
    } catch { /* transient poll error, keep trying */ }
  }, 1000)
}

onBeforeUnmount(() => clearInterval(pollTimer))

async function testLLM() {
  testing.value = true
  try {
    const r = await api.testLLM(settings.value.llm)
    if (r.ok) ElMessage.success('连接成功，模型回复: ' + r.reply)
    else ElMessage.error('连接失败: ' + r.error)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    testing.value = false
  }
}
</script>

<template>
  <div v-if="settings">
    <el-card shadow="never" class="section">
      <template #header>🎨 界面主题</template>
      <div class="themes">
        <div
          v-for="t in THEMES" :key="t.value"
          class="theme-tile" :class="{ active: theme === t.value }"
          @click="pickTheme(t.value)"
        >
          <div class="theme-swatch" :style="{ background: t.swatch[0] }">
            <span class="chip" :style="{ background: t.swatch[1] }" />
            <span class="chip accent" :style="{ background: t.swatch[2] }" />
          </div>
          <div class="theme-name">
            {{ t.name }}
            <el-tag v-if="t.value === 'slate'" size="small" type="info">默认</el-tag>
          </div>
          <div class="theme-desc">{{ t.desc }}</div>
        </div>
      </div>
      <div class="hint" style="margin: 10px 0 0; display: block">
        点击即刻切换，无需保存；主题只存在当前浏览器（换设备/清缓存后回到默认）。
      </div>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>🤖 翻译模型（OpenAI 兼容 API）</template>
      <el-form label-width="150px">
        <el-form-item label="API 地址 (base_url)">
          <el-input v-model="settings.llm.base_url" placeholder="https://api.openai.com/v1" />
          <span class="hint">兼容 OpenAI / DeepSeek / 通义 / Kimi / Ollama (http://localhost:11434/v1) 等</span>
        </el-form-item>
        <el-form-item label="API Key">
          <el-input v-model="settings.llm.api_key" type="password" show-password />
        </el-form-item>
        <el-form-item label="模型名称">
          <el-input v-model="settings.llm.model" placeholder="gpt-4o-mini / deepseek-chat / …" />
        </el-form-item>
        <el-form-item label="视觉模型">
          <el-input v-model="settings.llm.vision_model" placeholder="（可选）qwen-vl-plus / gpt-4o-mini / glm-4v …" />
          <span class="hint">仅画面翻译功能使用，留空则用上方主模型；DeepSeek 等纯文本模型不支持画面翻译</span>
        </el-form-item>
        <el-form-item label="Temperature">
          <el-slider v-model="settings.llm.temperature" :min="0" :max="1.5" :step="0.1" show-input style="width: 400px" />
        </el-form-item>
        <el-form-item label="每批翻译行数">
          <el-input-number v-model="settings.llm.batch_size" :min="10" :max="300" :step="10" />
          <span class="hint">受模型单次输出上限约束，一般 50~120</span>
        </el-form-item>
        <el-form-item label="模型上下文上限">
          <el-input-number v-model="settings.llm.context_limit" :min="4000" :max="1000000" :step="1000" />
          <span class="hint">tokens；全片超出预算时自动切换分块翻译</span>
        </el-form-item>
        <el-form-item>
          <el-button :loading="testing" @click="testLLM">测试连接</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>🎙️ 语音识别（Faster Whisper）</template>
      <el-form label-width="150px">
        <el-form-item label="模型">
          <el-select v-model="settings.asr.model_size" style="width: 260px" @change="refreshModelStatus">
            <el-option v-for="m in WHISPER_MODELS" :key="m.value" :value="m.value" :label="m.value">
              <span>{{ m.value }}</span>
              <span class="model-size">{{ fmtModelSize(m.size) }}</span>
            </el-option>
          </el-select>
          <el-tag v-if="modelDownloaded === true" type="success" class="tag">已下载，可离线使用</el-tag>
          <el-tag v-else-if="modelDownloaded === false" type="warning" class="tag">未下载</el-tag>
          <el-button
            v-if="modelDownloaded === false && download.status !== 'downloading'"
            size="small" type="primary" plain class="tag" @click="startDownload"
          >
            立即下载
          </el-button>
          <el-progress
            v-if="download.status === 'downloading'"
            :percentage="Math.round(download.progress || 0)"
            style="width: 180px; margin-left: 12px"
          />
          <span class="hint">large-v3-turbo 速度快约 8 倍、质量略降；调试可用 tiny</span>
        </el-form-item>
        <el-form-item label="本地模型目录">
          <el-input
            v-model="settings.asr.model_path"
            placeholder="（可选）CTranslate2 模型文件夹路径，含 model.bin"
            @change="refreshModelStatus"
          />
          <span class="hint">填写后优先使用该目录并忽略上面的模型选择，完全离线；适合手动下载好的模型</span>
        </el-form-item>
        <el-form-item label="设备">
          <el-radio-group v-model="settings.asr.device">
            <el-radio value="cpu">CPU</el-radio>
            <el-radio value="cuda">CUDA (GPU)</el-radio>
            <el-radio value="auto">自动</el-radio>
          </el-radio-group>
          <el-tag v-if="cuda" :type="cuda.available ? 'success' : 'info'" class="tag">
            {{ cuda.available ? `检测到 ${cuda.device_count} 个 CUDA 设备` : '本机未检测到可用 CUDA' }}
          </el-tag>
          <span class="hint">GPU 需安装 CUDA 运行库：pip install nvidia-cublas-cu12 nvidia-cudnn-cu12</span>
        </el-form-item>
        <el-form-item label="计算精度">
          <el-select v-model="settings.asr.compute_type" style="width: 200px">
            <el-option v-for="c in COMPUTE_TYPES" :key="c" :value="c" :label="c" />
          </el-select>
          <span class="hint">CPU 推荐 int8，GPU 推荐 float16 或 int8_float16</span>
        </el-form-item>
        <el-form-item label="Beam Size">
          <el-input-number v-model="settings.asr.beam_size" :min="1" :max="10" />
        </el-form-item>
        <el-form-item label="词级时间戳">
          <el-switch v-model="settings.asr.word_timestamps" />
          <span class="hint">按每个词的真实时间切分字幕行，时间轴更准、可杜绝碎行（推荐开启，速度略降 10-20%）</span>
        </el-form-item>
        <el-form-item label="VAD 语音检测">
          <el-switch v-model="settings.asr.vad_filter" />
          <span class="hint">过滤无语音片段，减少幻听字幕</span>
        </el-form-item>
        <template v-if="settings.asr.vad_filter">
          <el-form-item label="VAD 预设">
            <div>
              <el-button
                v-for="p in VAD_PRESETS" :key="p.name"
                size="small" style="margin: 0 8px 4px 0"
                @click="applyVadPreset(p)"
              >
                {{ p.name }}
              </el-button>
              <div v-for="p in VAD_PRESETS" :key="'d-' + p.name" class="hint" style="margin: 0; display: block">
                <strong>{{ p.name }}</strong>：{{ p.desc }}
              </div>
            </div>
          </el-form-item>
          <el-form-item label="VAD 灵敏度阈值">
            <el-slider v-model="settings.asr.vad_threshold" :min="0.05" :max="0.95" :step="0.05" show-input style="width: 400px" />
            <span class="hint">默认 0.35；有台词被漏识别时调低，误把噪音当语音时调高</span>
          </el-form-item>
          <el-form-item label="语音前后填充">
            <el-input-number v-model="settings.asr.vad_speech_pad_ms" :min="0" :max="3000" :step="100" />
            <span class="hint">毫秒，默认 400；句首/句尾被切掉时增大（如 800）</span>
          </el-form-item>
          <el-form-item label="最短语音时长">
            <el-input-number v-model="settings.asr.vad_min_speech_ms" :min="0" :max="5000" :step="50" />
            <span class="hint">毫秒，默认 250；短促的感叹词被丢弃时调低</span>
          </el-form-item>
          <el-form-item label="最短静默时长">
            <el-input-number v-model="settings.asr.vad_min_silence_ms" :min="100" :max="10000" :step="100" />
            <span class="hint">毫秒，默认 2000；低于此时长的停顿不会切断语音段</span>
          </el-form-item>
        </template>
      </el-form>
      <div class="model-notes">
        <p><strong>📌 模型选择说明</strong>（列表右侧为下载体积，模型仅在首次选用时下载一次）</p>
        <p>· <strong>为什么默认 large-v2</strong>：large-v3 在安静的基准测试中略准，但在真实影视音频中幻觉率明显更高（第三方实测约为 v2 的 4 倍）——电影中大量的配乐、音效和静默正是幻觉的高发场景，会凭空产生不存在的台词。因此默认使用更稳定的 large-v2。</p>
        <p>· <strong>CrisperWhisper</strong>：针对幻觉和逐字转写强化的模型，能更忠实地转写每个词、词级时间戳更准。仅支持英语和德语影片。</p>
      </div>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>💾 存储</template>
      <el-form label-width="150px">
        <el-form-item label="临时工作文件夹">
          <el-input v-model="settings.work_dir" placeholder="（留空 = 系统缓存目录）" />
          <span class="hint">存放提取音频等中间文件；程序只管理其中的 jobs 子目录并在启动时清空，保存后新任务立即生效</span>
        </el-form-item>
        <el-form-item label="模型存储位置">
          <el-input v-model="settings.model_cache_dir" placeholder="（留空 = HuggingFace 默认缓存目录）" />
          <span class="hint">修改后新下载的模型存到新位置；已下载的模型不会自动迁移（原位置的模型会被视为未下载）</span>
        </el-form-item>
        <el-form-item v-if="storageInfo" label=" ">
          <div class="storage-path">
            📂 模型当前实际存放于：<code>{{ storageInfo.effective_dir }}</code>
            <el-tag v-if="storageInfo.is_default" size="small" type="info" style="margin-left: 8px">默认位置</el-tag>
          </div>
        </el-form-item>
        <el-form-item v-if="logInfo" label="日志文件夹">
          <div class="storage-path">
            📝 <code>{{ logInfo.dir }}</code>
            <span class="hint">每个任务一个文件，保留最近 20 个；清空缓存不会删除</span>
            <div v-if="logInfo.files.length" class="log-list">
              <div v-for="f in logInfo.files.slice(0, 5)" :key="f.name" class="log-item">
                <a :href="'/api/logs/file/' + encodeURIComponent(f.name)" download>{{ f.name }}</a>
                <span class="log-size">{{ (f.size / 1024).toFixed(0) }} KB</span>
              </div>
            </div>
            <span v-else class="hint" style="margin-left: 0">（暂无日志，运行一次任务后生成）</span>
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>🌐 网络</template>
      <el-form label-width="150px">
        <el-form-item label="HTTPS 代理地址">
          <el-input v-model="settings.network.proxy_url" placeholder="http://127.0.0.1:7890" style="width: 320px" />
          <span class="hint">留空则不使用代理，格式 http://主机:端口（如 Clash 默认 7890）</span>
        </el-form-item>
        <el-form-item label="LLM API 走代理">
          <el-switch v-model="settings.network.llm_via_proxy" />
          <span class="hint">翻译请求与「测试连接」经代理发送</span>
        </el-form-item>
        <el-form-item label="模型下载走代理">
          <el-switch v-model="settings.network.model_download_via_proxy" />
          <span class="hint">从 HuggingFace 下载语音识别模型时经代理（无法直连时开启）</span>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>📝 字幕</template>
      <el-form label-width="150px">
        <el-form-item label="每行最大字符数">
          <el-input-number v-model="settings.subtitle.max_chars_per_line" :min="10" :max="120" />
        </el-form-item>
        <el-form-item label="单条最大时长（秒）">
          <el-input-number v-model="settings.subtitle.max_duration" :min="1" :max="15" :step="0.5" />
        </el-form-item>
        <el-form-item label="双语排版">
          <el-radio-group v-model="settings.subtitle.bilingual_layout">
            <el-radio value="translation_bottom">译文在下</el-radio>
            <el-radio value="translation_top">译文在上</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="字幕样式">
          <el-switch v-model="settings.subtitle.style_enabled" />
          <span class="hint">开启后输出 .ass 格式（支持字号/颜色，主流播放器均可自动挂载）；关闭输出标准 .srt</span>
        </el-form-item>
        <template v-if="settings.subtitle.style_enabled">
          <el-form-item label="译文字号">
            <el-slider v-model="settings.subtitle.font_size" :min="24" :max="100" show-input style="width: 400px" />
            <span class="hint">按 1080P 画布计算</span>
          </el-form-item>
          <el-form-item label="原文字号">
            <el-slider v-model="settings.subtitle.original_font_size" :min="16" :max="100" show-input style="width: 400px" />
          </el-form-item>
          <el-form-item label="译文颜色">
            <el-color-picker v-model="settings.subtitle.translation_color" />
            <span class="hint" style="margin-right: 24px">{{ settings.subtitle.translation_color }}</span>
            <span style="margin-right: 8px">原文颜色</span>
            <el-color-picker v-model="settings.subtitle.original_color" />
            <span class="hint">{{ settings.subtitle.original_color }}</span>
          </el-form-item>
          <el-form-item label="效果预览">
            <div class="subtitle-preview">
              <template v-if="settings.subtitle.bilingual_layout === 'translation_top'">
                <div :style="transStyle">不要问你的国家能为你做什么</div>
                <div :style="origStyle">ask not what your country can do for you</div>
              </template>
              <template v-else>
                <div :style="origStyle">ask not what your country can do for you</div>
                <div :style="transStyle">不要问你的国家能为你做什么</div>
              </template>
            </div>
          </el-form-item>
        </template>
      </el-form>
    </el-card>

    <el-button type="primary" size="large" :loading="saving" @click="save">保存设置</el-button>
  </div>
  <el-skeleton v-else :rows="8" animated />
</template>

<style scoped>
.section {
  margin-bottom: 16px;
}
.hint {
  margin-left: 12px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.tag {
  margin-left: 12px;
}
.model-size {
  float: right;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.model-notes {
  margin-top: 4px;
  padding: 10px 14px;
  background: var(--app-note-bg);
  border-radius: var(--app-radius);
  font-size: 12.5px;
  line-height: 1.8;
  color: var(--el-text-color-regular);
}
.themes {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.theme-tile {
  width: 208px;
  padding: 10px;
  border: 1px solid var(--el-border-color-light);
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.theme-tile:hover {
  border-color: var(--el-color-primary-light-5);
}
.theme-tile.active {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 1px var(--el-color-primary);
}
.theme-swatch {
  height: 52px;
  border-radius: calc(var(--app-radius) - 4px);
  border: 1px solid var(--el-border-color-lighter);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 10px;
}
.theme-swatch .chip {
  width: 46px;
  height: 26px;
  border-radius: calc(var(--app-radius) - 6px);
  border: 1px solid rgba(128, 128, 128, 0.25);
}
.theme-swatch .chip.accent {
  width: 26px;
}
.theme-name {
  margin-top: 8px;
  font-size: 13.5px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.theme-desc {
  margin-top: 2px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--el-text-color-secondary);
}
.model-notes p {
  margin: 2px 0;
}
.storage-path {
  font-size: 12.5px;
  color: var(--el-text-color-regular);
}
.subtitle-preview {
  width: 480px;
  height: 150px;
  border-radius: 6px;
  background: linear-gradient(160deg, #2c3e50 0%, #4a3f55 55%, #1a252f 100%);
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
  padding-bottom: 14px;
  line-height: 1.45;
  font-family: 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
}
.log-list {
  margin-top: 6px;
}
.log-item {
  display: flex;
  gap: 12px;
  align-items: baseline;
  padding: 2px 0;
  font-size: 12.5px;
}
.log-item a {
  color: var(--el-color-primary);
  text-decoration: none;
  font-family: var(--app-mono);
}
.log-item a:hover {
  text-decoration: underline;
}
.log-size {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.storage-path code {
  background: var(--el-fill-color);
  font-family: var(--app-mono);
  padding: 2px 6px;
  border-radius: 4px;
  word-break: break-all;
}
</style>

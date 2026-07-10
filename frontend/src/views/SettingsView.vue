<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const settings = ref(null)
const saving = ref(false)
const testing = ref(false)
const modelDownloaded = ref(null) // null = unknown / loading
const cuda = ref(null) // { available, device_count }
const download = ref({ status: 'idle', progress: 0 })
let pollTimer = null

// size = download size in MB (from the HuggingFace repos, model.bin + config)
const WHISPER_MODELS = [
  { value: 'tiny', size: 75 },
  { value: 'base', size: 141 },
  { value: 'small', size: 464 },
  { value: 'medium', size: 1460 },
  { value: 'large-v2', size: 2946 },
  { value: 'large-v3', size: 2948 },
  { value: 'large-v3-turbo', size: 1547 },
  { value: 'distil-large-v3', size: 1446 },
  { value: 'kotoba-whisper-v2.0', size: 1446 },
  { value: 'CrisperWhisper', size: 2948 },
]

function fmtModelSize(mb) {
  return mb >= 1000 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB'
}
const COMPUTE_TYPES = ['int8', 'int8_float16', 'float16', 'float32']

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
  } catch (e) {
    ElMessage.error('加载设置失败: ' + e.message)
  }
})

async function save() {
  saving.value = true
  try {
    settings.value = await api.saveSettings(settings.value)
    ElMessage.success('设置已保存')
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
        <el-form-item label="VAD 语音检测">
          <el-switch v-model="settings.asr.vad_filter" />
          <span class="hint">过滤无语音片段，减少幻听字幕</span>
        </el-form-item>
      </el-form>
      <div class="model-notes">
        <p><strong>📌 模型选择说明</strong>（列表右侧为下载体积，模型仅在首次选用时下载一次）</p>
        <p>· <strong>为什么默认 large-v2</strong>：large-v3 在安静的基准测试中略准，但在真实影视音频中幻觉率明显更高（第三方实测约为 v2 的 4 倍）——电影中大量的配乐、音效和静默正是幻觉的高发场景，会凭空产生不存在的台词。因此默认使用更稳定的 large-v2。</p>
        <p>· <strong>kotoba-whisper-v2.0</strong>：日语专用蒸馏模型，日语准确率不低于 large-v2、接近 large-v3，速度约为其 6 倍且无 v3 的幻觉问题。翻译日语影片时建议优先选用。</p>
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
  color: #909399;
  font-size: 12px;
}
.tag {
  margin-left: 12px;
}
.model-size {
  float: right;
  color: #909399;
  font-size: 12px;
}
.model-notes {
  margin-top: 4px;
  padding: 10px 14px;
  background: #f5f7fa;
  border-radius: 6px;
  font-size: 12.5px;
  line-height: 1.8;
  color: #606266;
}
.model-notes p {
  margin: 2px 0;
}
</style>

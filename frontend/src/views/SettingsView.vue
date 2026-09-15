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
const testingVision = ref(false)
const testingAsrApi = ref(false)
const modelDownloaded = ref(null) // null = unknown / loading
const cuda = ref(null) // { available, device_count }
const storageInfo = ref(null) // { effective_dir, is_default }
const logInfo = ref(null) // { dir, files: [{name, size, modified}] }
const serverInfo = ref(null) // { configured, running, lan_ips, urls, mcp, token }
const download = ref({ status: 'idle', progress: 0 })
let pollTimer = null

// Binding is read once at startup, so a saved change only lands on the next
// launch. Comparing saved against actually-running is the only honest way to
// say so — the alternative is a banner that lies after the user restarts.
const bindingPending = computed(() => {
  const info = serverInfo.value
  if (!info || !info.running) return false // started by an external command
  const wantHost = info.configured.lan_access ? '0.0.0.0' : '127.0.0.1'
  return info.running.host !== wantHost || info.running.port !== info.configured.port
})

const mcpConfigSnippet = computed(() => {
  const info = serverInfo.value
  if (!info) return ''
  const url = info.urls.mcp[0] || info.urls.mcp_local
  const headers = info.token
    ? `,\n      "headers": { "Authorization": "Bearer ${info.token}" }`
    : ''
  return `{
  "mcpServers": {
    "movietranslator": {
      "type": "http",
      "url": "${url}"${headers}
    }
  }
}`
})

async function refreshServerInfo() {
  try {
    serverInfo.value = await api.serverInfo()
  } catch {
    serverInfo.value = null
  }
}

async function copy(text, what = '已复制') {
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success(what)
  } catch {
    // clipboard access needs a secure context; over plain http on a LAN
    // address the browser refuses, and silently failing looks like a bug
    ElMessage.warning('浏览器不允许自动复制，请手动选中后复制')
  }
}

async function regenerateToken() {
  try {
    await api.regenerateToken()
    await refreshServerInfo()
    ElMessage.success('已生成新令牌，已连接的其他设备需要用新链接重新打开')
  } catch (e) {
    ElMessage.error('生成失败: ' + e.message)
  }
}

// size = download size in MB (from the HuggingFace repos, model.bin + config)
// VAD presets: [threshold, speech_pad_ms, min_speech_ms, min_silence_ms]
const VAD_PRESETS = [
  { name: '宽松·防漏（默认）', desc: '适合大多数电影：阈值 0.35 对偏小声的对白也能捕捉，最短语音 100ms 保住短促的应答词', v: [0.35, 400, 100, 2000] },
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

// One control for what used to be two. "Which engine" and "which device"
// are not independent choices to a user: picking the API engine means no
// device at all, and the device radio was hidden in that case anyway. So
// they read as one list — CPU / GPU / 自动 / API — while the settings
// underneath stay exactly as they were. Leaving api selected keeps the
// last local device, so switching back does not silently land on CPU.
const asrTarget = computed(() =>
  settings.value?.asr.engine === 'api' ? 'api' : settings.value?.asr.device
)

function setAsrTarget(value) {
  if (value === 'api') {
    settings.value.asr.engine = 'api'
    return
  }
  settings.value.asr.engine = 'local'
  settings.value.asr.device = value
}

// which recognition model reads the graphic subtitles (services/ocr.py)
const OCR_LANGS = [
  { value: '', label: '自动判定' },
  { value: 'en', label: '英语' },
  { value: 'ja', label: '日语' },
  { value: 'zh', label: '中文' },
  { value: 'ko', label: '韩语' },
  { value: 'ru', label: '俄语' },
  { value: 'fr', label: '法语 / 其它拉丁字母' },
]

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
    if (settings.value.asr.engine === 'local') {
      // the API engine loads no model and touches no GPU; asking would only
      // put a 「未下载」 tag next to something it never uses
      refreshModelStatus()
      api.cudaStatus().then((r) => (cuda.value = r)).catch(() => {})
    }
    api.storageInfo().then((r) => (storageInfo.value = r)).catch(() => {})
    api.logs().then((r) => (logInfo.value = r)).catch(() => {})
    refreshServerInfo()
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
    if (settings.value.asr.engine === 'local') refreshModelStatus()
    refreshServerInfo()
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
    if (!r.ok) {
      ElMessage.error('连接失败: ' + r.error)
    } else {
      ElMessage.success('连接成功，模型回复: ' + r.reply)
      if (r.thinking) {
        // shown separately and left on screen longer: whether the switch
        // actually took effect is the thing being tested here
        ElMessage({ type: r.thinking.level, message: r.thinking.text, duration: 6000 })
      }
    }
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    testing.value = false
  }
}

async function testAsrApi() {
  testingAsrApi.value = true
  try {
    const r = await api.testAsrApi(settings.value.llm)
    const at = `${r.model} @ ${r.endpoint}`
    if (!r.ok) {
      ElMessage({ type: 'error', duration: 8000, message: `语音接口连接失败（${at}）：${r.error}` })
    } else if (r.carried_audio === false) {
      // the reply can look perfect while the audio never arrived: only the
      // token count the server reports gives that away
      ElMessage({
        type: 'warning', duration: 10000,
        message: `接口通了（${at}），但服务端只收到了文字的 token 量`
          + `（${r.prompt_tokens}，带音频应约 ${r.expected_tokens}）——`
          + '中转多半把音频部分丢掉了，这样识别出来的内容全是编的',
      })
    } else if (r.heard_it) {
      ElMessage.success(`语音接口可用（${at}），正确听出了测试音的音调走向（${r.asked}）`)
    } else {
      ElMessage({
        type: 'warning', duration: 8000,
        message: `接口通了（${at}），但模型没听出测试音是${r.asked}的，回复是「${r.reply}」——`
          + '多半是这个模型不支持音频输入',
      })
    }
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    testingAsrApi.value = false
  }
}

async function testVision() {
  testingVision.value = true
  try {
    const r = await api.testVision(settings.value.llm)
    const at = `${r.model} @ ${r.endpoint}`
    if (!r.ok) {
      ElMessage({ type: 'error', duration: 8000, message: `视觉模型连接失败（${at}）：${r.error}` })
    } else if (r.read_it) {
      ElMessage.success(`视觉模型可用（${at}），已正确读出测试图片里的文字`)
    } else {
      // it answered, so the endpoint is fine — but it did not read the
      // picture, which is the only thing this model is here to do
      ElMessage({
        type: 'warning', duration: 8000,
        message: `接口通了（${at}），但模型没能读出测试图片里的文字，回复是「${r.reply}」——`
          + '多半是这个模型不支持图像输入，或服务端没按视觉模型加载它',
      })
    }
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    testingVision.value = false
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
          <span class="hint">
            画面翻译、以及「图形字幕 OCR」选用视觉引擎时使用；留空则用上方主模型。
            DeepSeek 等纯文本模型不支持这两项。
          </span>
        </el-form-item>
        <el-form-item label="视觉 API 地址">
          <el-input v-model="settings.llm.vision_base_url" placeholder="（可选）http://127.0.0.1:1234/v1" />
          <span class="hint">
            视觉模型单独的 base_url，留空＝与上方主接口相同。本地跑 VL 模型、翻译走云端时填这里——
            两者在同一个任务里先后使用，中途改不了设置。<strong>注意结尾的 /v1 不能少。</strong>
          </span>
        </el-form-item>
        <el-form-item label="视觉 API Key">
          <el-input v-model="settings.llm.vision_api_key" type="password" show-password
                    placeholder="（可选）本地服务通常不需要" />
          <span class="hint">仅在填了上面那个地址时使用；主模型的 Key 不会被发往另一个地址</span>
        </el-form-item>
        <el-form-item label="语音模型">
          <el-input v-model="settings.llm.audio_model" placeholder="（可选）支持音频输入的多模态模型" />
          <span class="hint">
            「语音识别」的引擎选为 API 时使用；留空则用上方主模型。
            必须是能接收音频输入的多模态模型，纯文本模型不行。
          </span>
        </el-form-item>
        <el-form-item label="语音 API 地址">
          <el-input v-model="settings.llm.audio_base_url" placeholder="（可选）http://127.0.0.1:1234/v1" />
          <span class="hint">
            留空＝与上方主接口相同。识别在任务开头、翻译在任务结尾，中途改不了设置，
            所以两者可以指向不同的服务商。<strong>注意结尾的 /v1 不能少。</strong>
          </span>
        </el-form-item>
        <el-form-item label="语音 API Key">
          <el-input v-model="settings.llm.audio_api_key" type="password" show-password
                    placeholder="（可选）本地服务通常不需要" />
          <span class="hint">仅在填了上面那个地址时使用；主模型的 Key 不会被发往另一个地址</span>
        </el-form-item>
        <el-form-item label="关闭思考模式">
          <el-switch v-model="settings.llm.disable_thinking" />
          <span class="hint">
            字幕整理和逐行翻译都是机械任务，模型的思考过程属于纯开销——实测转写预处理的
            输出 token 是它需要重述内容的 5～6 倍。而且 DeepSeek 文档说明思考模式会
            <strong>使 temperature 失效</strong>，本程序给预处理设定的 temperature=0 只有关掉思考才生效。
            <br />不支持该参数的服务会被自动识别并跳过，不影响使用。
          </span>
        </el-form-item>
        <el-form-item label="Temperature">
          <el-slider v-model="settings.llm.temperature" :min="0" :max="1.5" :step="0.1" show-input style="width: 400px" />
        </el-form-item>
        <el-form-item label="每批翻译行数">
          <el-input-number v-model="settings.llm.batch_size" :min="10" :max="500" :step="10" />
          <span class="hint">
            默认 200。调大可减少请求次数、缩短耗时并降低 token 消耗；
            受模型单次输出上限约束，若出现响应被截断（日志报「行号覆盖校验未通过」或反复补翻）则调小。
          </span>
        </el-form-item>
        <el-form-item label="模型上下文上限">
          <el-input-number v-model="settings.llm.context_limit" :min="4000" :max="1000000" :step="1000" />
          <span class="hint">tokens；全片超出预算时自动切换分块翻译</span>
        </el-form-item>
        <el-form-item>
          <el-button :loading="testing" @click="testLLM">测试连接</el-button>
          <el-button :loading="testingVision" @click="testVision">测试视觉模型</el-button>
          <el-button :loading="testingAsrApi" @click="testAsrApi">测试语音模型</el-button>
          <span class="hint">
            视觉测试会发一张写着字的图片过去，要求模型读出来——纯文本模型能答完文字测试，
            却会在第一条字幕上失败。语音测试会发一段音调变化的测试音过去要求听出方向，
            并核对服务端收到的 token 量：中转把音频丢掉时，模型照样会答得头头是道。
          </span>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>🎙️ 语音识别</template>
      <el-form label-width="150px">
        <el-form-item label="识别设备">
          <el-radio-group :model-value="asrTarget" @update:model-value="setAsrTarget">
            <el-radio value="cpu">CPU</el-radio>
            <el-radio value="cuda">CUDA (GPU)</el-radio>
            <el-radio value="auto">自动</el-radio>
            <el-radio value="api">API（多模态大模型）</el-radio>
          </el-radio-group>
          <el-tag v-if="cuda && asrTarget !== 'api'" :type="cuda.available ? 'success' : 'info'" class="tag">
            {{ cuda.available ? `检测到 ${cuda.device_count} 个 CUDA 设备` : '本机未检测到可用 CUDA' }}
          </el-tag>
          <div class="hint" style="margin: 4px 0 0; display: block">
            <template v-if="settings.asr.engine === 'local'">
              在本机运行 Faster Whisper，音频不出本机，不花钱。需要下载模型（下方），
              大模型在 CPU 上很慢。<br />
              GPU 需安装 CUDA 运行库：pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
            </template>
            <template v-else>
              <strong>音频会分段上传</strong>到「翻译模型」一栏里配置的「语音模型」接口——
              这是本软件唯一会上传音频的情况。不下模型、不占 GPU，按 token 计费
              （音频约 25 tokens/秒，两小时影片约 18 万输入 token）。<br />
              时间轴由本机决定：先用本地语音检测在静音处切段，模型只在每段内部计时，
              所以它的时钟就算不准也不会越走越偏。<br />
              代价：<strong>没有词级时间戳</strong>（断句精度略降），同一段音频两次识别
              结果不会完全一样，且「二次识别」对本引擎不适用。<br />
              <strong>目前只针对 gemini-3.8-flash-high 做过调优</strong>：窗口长度、
              时间戳格式、标点要求、失败重试都是按它的实测行为定的。换用别的模型仍然可以跑，
              但这些参数未必合适，建议先用下方「测试语音模型」确认音频真的送达、方向题答对。
            </template>
          </div>
        </el-form-item>
        <template v-if="settings.asr.engine === 'local'">
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
        <el-form-item label="二次识别">
          <el-switch v-model="settings.asr.second_pass" />
          <span class="hint">
            语音检测判定为「无语音」的段落，关闭检测后再识别一遍，只保留通过质量校验的结果。
            对白压在配乐下的片源（纪录片、恐怖片、综艺）常被整段漏掉——实测一部日本恐怖片
            9 分钟的剧情段落一句都没识别到，开启后找回了完整对话。
            <br />找回的内容会再交给 AI 逐条复核是否属于本片——语音识别模型在没有人声的地方
            会吐出与影片无关但语句通顺的内容（片尾语、车站广播、烹饪教程），复核会把它们丢弃。
            AI 复核失败时，本次找回的内容一律不采用，保证结果不会比关闭本项更差。
            <br />代价：识别时间约翻倍，且因为文字变多，后续翻译的 token 消耗也会同比增加。
            片源对白清晰、字幕数量正常时可关闭。
          </span>
        </el-form-item>
        <el-form-item label="分窗识别兜底">
          <el-switch v-model="settings.asr.windowed_first_pass" />
          <span class="hint">
            给<strong>严重劣化</strong>的片源准备的——不是「模拟片源」这么宽，实测两盘普通的 VHS 转录
            语音检测都正常工作。真正会失灵的是劣化到人耳都听得费劲的采集：一盘 96 分钟的带子里
            检测只认出 5 分钟是「语音」，于是第一遍几乎什么都没识别，全靠二次识别去捡，
            而复核又没有足够的已确认台词可以对照。
            <br />开启后，遇到这种片源时第一遍改成「把整条时间轴切成 5 分钟一段、关掉语音检测逐段识别」，
            语音检测也听得到的那些行照旧算第一遍结果，只有关掉检测才看得见的行才送 AI 复核——
            删除权限和不开本项时完全一样。
            <br /><strong>只在语音检测保留不到 10% 且检测之外确实还有接近人声音量的声音时才会触发</strong>，
            数字片源（BD/DVD）永远不会触发，空音轨或选错音轨也不会。默认关闭。
          </span>
        </el-form-item>
        </template>
        <template v-else>
        <el-form-item label="每段时长">
          <el-input-number v-model="settings.asr.api_window_seconds" :min="60" :max="420" :step="30" />
          <span class="hint">
            秒，默认 300。每段单独发一次请求，段内时间由模型给、段的起点由本机给。
            上限 420 秒不是为了省钱：多模态模型的时间戳会随音频变长而越走越偏，
            段太长它自己的时间就不能用了。
          </span>
        </el-form-item>
        <el-form-item label="音频格式">
          <el-radio-group v-model="settings.asr.api_audio_format">
            <el-radio value="mp3">mp3</el-radio>
            <el-radio value="wav">wav</el-radio>
          </el-radio-group>
          <span class="hint">mp3 体积只有 wav 的八分之一，两者服务端都接受；接口拒收 mp3 时才改 wav</span>
        </el-form-item>
        <el-form-item label="同时请求数">
          <el-input-number v-model="settings.asr.api_concurrency" :min="1" :max="8" />
          <span class="hint">几段同时发。接口限流（429）时调小；两小时影片约 24 段</span>
        </el-form-item>
        <el-form-item label="最短语音时长">
          <el-input-number v-model="settings.asr.vad_min_speech_ms" :min="0" :max="5000" :step="50" />
          <span class="hint">毫秒；本引擎只用它来判断哪里是停顿、可以切段</span>
        </el-form-item>
        <el-form-item label="最短静默时长">
          <el-input-number v-model="settings.asr.vad_min_silence_ms" :min="100" :max="10000" :step="100" />
          <span class="hint">毫秒；同上。切段永远切在静音处，不会把一句话切成两半</span>
        </el-form-item>
        </template>
        <el-form-item label="识别提示词">
          <el-input
            v-model="settings.asr.initial_prompt"
            type="textarea" :rows="2"
            placeholder="例：藤堂、亜希子、柳川、樺山、人狼ゲーム"
          />
          <span class="hint">
            把片中反复出现的人名/专有名词写在这里，能显著减少人名识别错误（如「藤堂」被听成「どうぞ」）。
            <strong>必须用影片的原始语言书写</strong>——写成中文会把整篇转写结果带偏，因此剧情简介和译名对照表不会自动用作提示词。
          </span>
        </el-form-item>
        <el-form-item v-if="settings.asr.engine === 'local'" label="标点示例句">
          <el-switch v-model="settings.asr.style_prompt" />
          <span class="hint">
            在识别提示词前面自动加一句带标点、带大小写的源语言示范句。语音识别模型会朝着提示词的样子写，
            而它经常整片不给句号——实测两部日语片有 72% / 57% 的行没有句末标点，断句只能整体推迟到转写预处理去做。
            <br />源语言选「自动」时会先做一次语言预检测，只用来挑示范句，不影响识别本身对语言的判断；
            没有对应示范句的语言不加。把示范句原样念回来的行会被丢弃。默认关闭。
          </span>
        </el-form-item>
        <el-form-item v-if="settings.asr.engine === 'local'" label="VAD 语音检测">
          <el-switch v-model="settings.asr.vad_filter" />
          <span class="hint">过滤无语音片段，减少幻听字幕</span>
        </el-form-item>
        <template v-if="settings.asr.engine === 'local' && settings.asr.vad_filter">
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
            <span class="hint">毫秒，默认 100；短促的感叹词（「诶」「嗯」）被整句丢弃时调低</span>
          </el-form-item>
          <el-form-item label="最短静默时长">
            <el-input-number v-model="settings.asr.vad_min_silence_ms" :min="100" :max="10000" :step="100" />
            <span class="hint">毫秒，默认 2000；低于此时长的停顿不会切断语音段</span>
          </el-form-item>
        </template>
      </el-form>
      <div class="model-notes">
        <p>· <strong>选 API 引擎时，下面这些都不生效</strong>：模型、设备、计算精度、Beam、词级时间戳、
          二次识别、VAD 阈值与前后填充。本机只负责切段，识别全在服务端。</p>
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
        <el-form-item label="调试模式">
          <el-switch v-model="settings.debug_mode" />
          <span class="hint">
            开启后，每个任务会在<strong>字幕输出的同一文件夹</strong>生成一个
            <code>视频名.debug.log</code>：语音识别的原始结果与词级时间戳、
            每一步断句/合并的判定与被否决的原因、以及全部 AI 请求与回复。
            字幕出现断句错乱、半个词、漏译等问题时开启它，把该文件发给开发者即可定位。
            <br />文件较大（两小时影片约数 MB），平时建议关闭。不含 API key。
          </span>
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
      <template #header>
        📡 局域网访问与 MCP 服务
        <el-tag v-if="bindingPending" type="warning" size="small" class="tag">改动需重启程序后生效</el-tag>
      </template>
      <el-form label-width="150px">
        <el-form-item label="局域网访问">
          <el-switch v-model="settings.server.lan_access" />
          <span class="hint">默认关闭，仅本机可访问</span>
          <div class="hint" style="margin: 4px 0 0; display: block; line-height: 1.7">
            开启后本程序会监听所有网卡，手机、笔记本等同网段设备可以直接打开网页发起翻译。
            <strong>修改后需要重启程序才会生效。</strong>
            <br />⚠️ 本程序的接口默认假定使用者就是本机用户：可以浏览<strong>本机任意目录</strong>、
            对任意路径发起任务、读取任务日志（含片源路径）。因此开启后请保持下面的访问令牌开启。
            <br />⚠️ 只提供明文 HTTP，没有加密。<strong>不要把端口直接转发到公网</strong>——
            需要在外网使用时，请套一层反向代理（Caddy / Nginx）或用 Tailscale、Cloudflare Tunnel 之类的隧道。
          </div>
        </el-form-item>
        <el-form-item label="监听端口">
          <el-input-number v-model="settings.server.port" :min="1" :max="65535" />
          <span class="hint">默认 8760，同样需要重启生效</span>
        </el-form-item>
        <el-form-item label="需要访问令牌">
          <el-switch v-model="settings.server.require_token" />
          <span class="hint">
            本机（127.0.0.1）<strong>始终免验证</strong>，不会把自己锁在外面；
            关闭后同网段任何人都能直接使用，仅建议在完全可信的网络里关闭
          </span>
        </el-form-item>
        <template v-if="serverInfo">
          <el-form-item v-if="serverInfo.token" label="访问令牌">
            <el-input :model-value="serverInfo.token" readonly style="width: 320px" />
            <el-button size="small" class="tag" @click="copy(serverInfo.token, '令牌已复制')">复制</el-button>
            <el-button size="small" class="tag" @click="regenerateToken">重新生成</el-button>
            <span class="hint">重新生成会让已放行的设备全部失效</span>
          </el-form-item>
          <el-form-item v-if="settings.server.lan_access" label="其他设备访问">
            <div style="width: 100%">
              <div v-if="!serverInfo.lan_ips.length" class="hint" style="margin-left: 0">
                （未检测到局域网地址，请确认本机已连接网络）
              </div>
              <div v-for="url in serverInfo.urls.lan" :key="url" class="url-row">
                <code>{{ url }}</code>
                <el-button size="small" text type="primary" @click="copy(url, '链接已复制')">复制</el-button>
              </div>
              <div class="hint" style="margin: 4px 0 0; display: block">
                在别的设备上用这条<strong>带令牌的链接</strong>打开一次即可，之后该设备无需再带令牌。
              </div>
            </div>
          </el-form-item>
        </template>

        <el-divider />

        <el-form-item label="MCP 服务">
          <el-switch v-model="settings.mcp.enabled" :disabled="serverInfo && !serverInfo.mcp.available" />
          <span class="hint">默认关闭，开关立即生效无需重启</span>
          <el-tag v-if="serverInfo && !serverInfo.mcp.available" type="danger" size="small" class="tag">
            依赖未安装，请重新执行 pip install -e .
          </el-tag>
          <div class="hint" style="margin: 4px 0 0; display: block; line-height: 1.7">
            开启后，Claude 等支持 MCP 的客户端可以直接驱动本机完成翻译：列出目录里的视频、
            选音轨、发起单片或整目录翻译、查询进度、取回字幕全文与任务日志。
            <br />任务在后台运行，发起后立刻返回任务号，客户端自行轮询进度——一部影片要跑几十分钟到几小时。
            <br /><strong>不提供修改设置的能力</strong>：识别模型、API key 等只能在本页改，远程无法改动。
            <br />在局域网/公网使用时，MCP 与网页共用上面的访问令牌（请求头 <code>Authorization: Bearer …</code>）。
          </div>
        </el-form-item>
        <template v-if="serverInfo && settings.mcp.enabled">
          <el-form-item label="MCP 地址">
            <div style="width: 100%">
              <div class="url-row">
                <code>{{ serverInfo.urls.mcp_local }}</code>
                <span class="hint">本机</span>
              </div>
              <div v-for="url in serverInfo.urls.mcp" :key="url" class="url-row">
                <code>{{ url }}</code>
                <el-button size="small" text type="primary" @click="copy(url, '地址已复制')">复制</el-button>
              </div>
            </div>
          </el-form-item>
          <el-form-item label="客户端配置">
            <div style="width: 100%">
              <pre class="snippet">{{ mcpConfigSnippet }}</pre>
              <el-button size="small" @click="copy(mcpConfigSnippet, '配置已复制')">复制配置</el-button>
              <span class="hint">
                Claude Code 可直接执行：
                <code>claude mcp add --transport http movietranslator {{ serverInfo.urls.mcp[0] || serverInfo.urls.mcp_local }}</code>
              </span>
            </div>
          </el-form-item>
        </template>
      </el-form>
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>🔤 图形字幕 OCR</template>
      <el-form label-width="150px">
        <el-form-item label="识别引擎">
          <el-radio-group v-model="settings.ocr.engine">
            <el-radio value="rapidocr">本地 OCR（RapidOCR）</el-radio>
            <el-radio value="vision">视觉大模型</el-radio>
          </el-radio-group>
          <div class="hint" style="margin: 4px 0 0; display: block">
            <template v-if="settings.ocr.engine === 'rapidocr'">
              本地运行、不联网、不花钱，一部片几十秒。首次使用会自动下载约 10–20MB 的模型。<br>
              需要先安装 OCR 组件：<code>pip install -e ".[ocr]"</code>
              （Windows 为 <code>.venv\Scripts\pip</code>，Linux 为 <code>.venv/bin/pip</code>）。
            </template>
            <template v-else>
              走上面「视觉模型」那一栏配置的模型（留空则用主模型），要联网、按 token 计费。
              日文的识别质量通常更好，但一部片会发出几十到上百次请求。
            </template>
          </div>
        </el-form-item>
        <el-form-item label="识别语言">
          <el-select v-model="settings.ocr.language" style="width: 200px">
            <el-option v-for="l in OCR_LANGS" :key="l.value" :value="l.value" :label="l.label" />
          </el-select>
          <span class="hint">留空则先看字幕轨的语言标签，没有标签时试认几条再判定</span>
        </el-form-item>
        <el-form-item label="放大倍数">
          <el-input-number v-model="settings.ocr.upscale" :min="1" :max="4" />
          <span class="hint">识别前把字幕图放大几倍；DVD（720×480）这类小图调到 3 会更准，1080p 用 2 即可</span>
        </el-form-item>
        <el-form-item v-if="settings.ocr.engine === 'vision'" label="每批条数">
          <el-input-number v-model="settings.ocr.vision_batch" :min="1" :max="40" />
          <span class="hint">一次请求拼几条字幕。越大越省钱越快，但模型数错行号时整批作废重来</span>
        </el-form-item>
      </el-form>
      <div class="model-notes">
        <p><strong>📌 什么时候会用到</strong></p>
        <p>· 蓝光原盘的字幕（PGS）、DVD 的字幕（VobSub）里存的是<strong>图片</strong>不是文字，
          必须先 OCR 才能翻译。首页「原文来源」选「片源已有的字幕」并选中这类轨道时自动启用。</p>
        <p>· OCR 的结果会再过一道<strong>同语言校对</strong>（复用「转写预处理」开关），
          专门纠正口/ロ、力/カ、rn/m 这类形近字误识别——这是握有全片上下文的模型最擅长的事。</p>
      </div>
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
.url-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 2px 0;
}
.url-row code {
  background: var(--el-fill-color);
  font-family: var(--app-mono);
  font-size: 12.5px;
  padding: 3px 8px;
  border-radius: 4px;
  word-break: break-all;
}
.snippet {
  margin: 0 0 8px;
  padding: 10px 12px;
  background: var(--app-note-bg);
  border-radius: var(--app-radius);
  font-family: var(--app-mono);
  font-size: 12px;
  line-height: 1.6;
  color: var(--el-text-color-primary);
  white-space: pre;
  overflow-x: auto;
}
.storage-path code {
  background: var(--el-fill-color);
  font-family: var(--app-mono);
  padding: 2px 6px;
  border-radius: 4px;
  word-break: break-all;
}
</style>

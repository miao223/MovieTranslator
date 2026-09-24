<script setup>
// The server-side file picker, shared by 翻译任务 and 原盘.
//
// mode decides what a click picks:
//   'file' — a video / audio / subtitle file
//   'dir'  — a folder, through the footer button
//   'disc' — a disc: a folder holding BDMV / VIDEO_TS (clicked directly, or
//            the footer button once inside one), or an .iso image
//   'discs' — a folder whose discs the 原盘 batch mode lists, through the
//            footer button; discs and images are marked but only browsed
// The listing itself comes from /api/fs/browse; mode="disc" makes the
// server list images and name the disc folders, so this component never
// needs its own copy of what a disc looks like.
import { reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  mode: { type: String, default: 'file' },
})
const emit = defineEmits(['update:modelValue', 'pick'])

const TITLES = {
  file: '选择视频 / 音频 / 字幕文件',
  dir: '选择目录',
  disc: '选择原盘（BDMV / VIDEO_TS 文件夹，或 .iso 镜像）',
  discs: '选择文件夹（里面的原盘会全部列出来）',
}
const EMPTY = {
  file: '此目录没有子目录或视频 / 音频文件',
  dir: '此目录没有子目录',
  disc: '此目录没有子目录或 .iso 镜像',
  discs: '此目录没有子目录或 .iso 镜像',
}

const browser = reactive({
  path: '', parent: null, dirs: [], files: [], disc_dirs: [], is_disc: false,
})
const addressInput = ref('')
const quickAccess = ref([])

async function open(path = '') {
  try {
    const data = await api.browse(path, discListing() ? 'disc' : '')
    Object.assign(browser, { disc_dirs: [], is_disc: false }, data)
    addressInput.value = data.path
    emit('update:modelValue', true)
    if (!quickAccess.value.length) {
      api.quickAccess().then((r) => (quickAccess.value = r.items)).catch(() => {})
    }
  } catch (e) {
    ElMessage.error(e.message)
  }
}

defineExpose({ open })

function joinPath(dir, name) {
  if (!dir) return name
  const sep = dir.includes('\\') ? '\\' : '/'
  return dir.endsWith(sep) ? dir + name : dir + sep + name
}

function fmtSize(bytes) {
  if (bytes > 1 << 30) return (bytes / (1 << 30)).toFixed(1) + ' GB'
  if (bytes > 1 << 20) return (bytes / (1 << 20)).toFixed(1) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

function choose(path) {
  emit('pick', path)
  emit('update:modelValue', false)
}

// the server marks discs and lists images for both disc modes
const discListing = () => props.mode === 'disc' || props.mode === 'discs'
const pickingFolder = () => props.mode === 'dir' || props.mode === 'discs'

function isDisc(name) {
  return discListing() && browser.disc_dirs.includes(name)
}

function clickDir(name) {
  const full = browser.path ? joinPath(browser.path, name) : name
  if (isDisc(name) && props.mode === 'disc') choose(full)
  else open(full)
}

function clickFile(f) {
  if (pickingFolder()) return
  choose(joinPath(browser.path, f.name))
}

// jump to a pasted Explorer path: a folder opens it, a file of the right
// kind is picked directly (quotes from "复制文件地址" are stripped server-side)
async function jumpToAddress() {
  const raw = addressInput.value.trim()
  if (!raw) return
  try {
    const r = await api.resolvePath(raw)
    if (r.type === 'dir') {
      open(r.path)
    } else if (r.type === 'file' && props.mode === 'disc') {
      // an .iso, or a file inside a disc (index.bdmv, an .IFO): the
      // analysis finds the disc from either and says so if there is none
      choose(r.path)
      ElMessage.success('已选择: ' + r.path)
    } else if (r.type === 'file' && r.is_media && props.mode === 'file') {
      choose(r.path)
      ElMessage.success('已选择: ' + r.path)
    } else if (r.type === 'file' && pickingFolder()) {
      ElMessage.warning('当前在选择目录，请粘贴文件夹路径或点「选择此目录」')
    } else if (r.type === 'file') {
      ElMessage.warning('该文件不是支持的视频 / 音频格式')
    } else {
      ElMessage.warning('路径不存在: ' + r.path)
    }
  } catch (e) {
    ElMessage.error(e.message)
  }
}

const fileIcon = (f) => ({ audio: '🎵', subtitle: '💬', disc: '💿' }[f.kind] || '🎬')
</script>

<template>
  <el-dialog
    :model-value="modelValue"
    :title="TITLES[mode] || TITLES.file"
    width="680px"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div class="browser-path">
      <el-button size="small" :disabled="browser.parent === null" @click="open(browser.parent)">
        ↑ 上级
      </el-button>
      <el-input
        v-model="addressInput"
        size="small"
        :placeholder="mode === 'disc'
          ? '粘贴原盘文件夹或 .iso 的完整路径，回车跳转'
          : pickingFolder() ? '粘贴文件夹的完整路径，回车跳转'
            : '粘贴文件夹或视频 / 音频文件的完整路径，回车跳转'"
        @keyup.enter="jumpToAddress"
      >
        <template #append>
          <el-button @click="jumpToAddress">跳转</el-button>
        </template>
      </el-input>
    </div>
    <div v-if="quickAccess.length" class="quick-access">
      <el-tag
        v-for="q in quickAccess" :key="q.path"
        class="quick-item" effect="plain" @click="open(q.path)"
      >
        {{ q.name }}
      </el-tag>
    </div>
    <div class="browser-list">
      <div
        v-for="d in browser.dirs" :key="'d-' + d"
        class="entry dir" :class="{ disc: isDisc(d) }" @click="clickDir(d)"
      >
        {{ isDisc(d) ? '💿' : '📁' }} {{ d }}
        <span v-if="isDisc(d)" class="size">{{ mode === 'disc' ? '原盘，点击选择' : '原盘' }}</span>
      </div>
      <div
        v-for="f in browser.files" :key="'f-' + f.name"
        class="entry file" @click="clickFile(f)"
      >
        {{ fileIcon(f) }} {{ f.name }} <span class="size">{{ fmtSize(f.size) }}</span>
      </div>
      <el-empty
        v-if="!browser.dirs.length && !browser.files.length"
        :description="EMPTY[mode] || EMPTY.file" :image-size="60"
      />
    </div>
    <template v-if="pickingFolder() || (mode === 'disc' && browser.is_disc)" #footer>
      <el-button @click="emit('update:modelValue', false)">取消</el-button>
      <el-button type="primary" :disabled="!browser.path" @click="choose(browser.path)">
        {{ mode === 'disc' ? '✓ 选择这张原盘' : '✓ 选择此目录' }}
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.browser-path {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.quick-access {
  margin-bottom: 8px;
}
.quick-item {
  margin: 0 6px 4px 0;
  cursor: pointer;
}
.browser-list {
  max-height: 380px;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-light);
  border-radius: var(--app-radius);
}
.entry {
  padding: 7px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.entry:hover {
  background: var(--el-color-primary-light-9);
}
.entry.disc {
  font-weight: 600;
}
.entry .size {
  float: right;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  font-weight: normal;
}
</style>

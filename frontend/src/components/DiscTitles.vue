<script setup>
// One disc's titles, with their ticks: the films, episodes, extras and other
// versions up front, the rest (duplicates, shorts, loops…) folded away.
// Shared by the 原盘 page's single-disc view and each disc of its batch
// view; the ticks belong to the page, this only shows and changes them.
import { computed, ref } from 'vue'

const props = defineProps({
  titles: { type: Array, required: true },
  picked: { type: Object, required: true },   // id → ticked
})
const emit = defineEmits(['toggle'])

const showRest = ref(false)
const PRIMARY = ['main', 'episode', 'extra', 'variant']
const TAG = { main: 'success', episode: 'success', extra: 'primary', variant: 'warning' }
const shown = computed(() => props.titles.filter((t) => PRIMARY.includes(t.category)))
const rest = computed(() => props.titles.filter((t) => !PRIMARY.includes(t.category)))

function tagType(t) {
  if (!t.selectable) return 'danger'
  return TAG[t.category] || 'info'
}

function hms(seconds) {
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

function fmtBytes(bytes) {
  if (bytes == null) return '—'
  if (bytes >= 1 << 30) return (bytes / (1 << 30)).toFixed(1) + ' GB'
  if (bytes >= 1 << 20) return (bytes / (1 << 20)).toFixed(0) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

function streamsOf(t, kind) {
  return t.streams.filter((s) => s.kind === kind)
}

function audioLine(s) {
  return [s.language_name || '未标注', s.codec, s.detail].filter(Boolean).join(' ')
    + (s.commentary ? '（评论）' : '')
}
</script>

<template>
  <el-table :data="shown" size="small" class="titles" row-key="id">
    <el-table-column width="46">
      <template #default="{ row }">
        <el-checkbox
          :model-value="picked[row.id]" :disabled="!row.selectable"
          @update:model-value="emit('toggle', row, $event)"
        />
      </template>
    </el-table-column>
    <el-table-column label="类型" width="100">
      <template #default="{ row }">
        <el-tag size="small" :type="tagType(row)">{{ row.label }}</el-tag>
      </template>
    </el-table-column>
    <el-table-column label="时长" width="84">
      <template #default="{ row }">{{ hms(row.duration) }}</template>
    </el-table-column>
    <el-table-column label="内容" min-width="300">
      <template #default="{ row }">
        <div class="streams">
          <span class="video">{{ row.video }}</span>
          <span v-if="row.chapters > 1"> · {{ row.chapters }} 章</span>
          <span v-if="row.angles > 1"> · {{ row.angles }} 个角度</span>
        </div>
        <div class="streams">
          音轨：<template v-for="(s, i) in streamsOf(row, 'audio')" :key="'a' + i">
            <span :class="{ dropped: !s.carried }" :title="s.note">{{ audioLine(s) }}</span><span v-if="i < streamsOf(row, 'audio').length - 1">；</span>
          </template>
        </div>
        <div v-if="streamsOf(row, 'subtitle').length" class="streams">
          字幕：<template v-for="(s, i) in streamsOf(row, 'subtitle')" :key="'s' + i">
            <span :class="{ dropped: !s.carried }" :title="s.note">{{ s.language_name || '未标注' }}{{ s.forced ? '（强制）' : '' }}</span><span v-if="i < streamsOf(row, 'subtitle').length - 1">、</span>
          </template>
        </div>
        <div class="reason">{{ row.reason }}</div>
        <div v-for="n in row.notes" :key="n" class="reason">· {{ n }}</div>
      </template>
    </el-table-column>
    <el-table-column label="大小" width="80">
      <template #default="{ row }">{{ fmtBytes(row.size) }}</template>
    </el-table-column>
    <el-table-column label="输出文件" min-width="200">
      <template #default="{ row }"><code class="out">{{ row.output }}</code></template>
    </el-table-column>
  </el-table>

  <div v-if="rest.length" class="rest">
    <el-link type="info" :underline="false" @click="showRest = !showRest">
      {{ showRest ? '▾' : '▸' }} 另外 {{ rest.length }} 个标题默认不导出（重复 / 过短 / 循环 / 无声 / 静态图），{{ showRest ? '收起' : '展开查看' }}
    </el-link>
    <el-table v-if="showRest" :data="rest" size="small" class="titles" row-key="id">
      <el-table-column width="46">
        <template #default="{ row }">
          <el-checkbox
            :model-value="picked[row.id]" :disabled="!row.selectable"
            @update:model-value="emit('toggle', row, $event)"
          />
        </template>
      </el-table-column>
      <el-table-column label="标题" width="130">
        <template #default="{ row }"><code>{{ row.id }}</code></template>
      </el-table-column>
      <el-table-column label="类型" width="80">
        <template #default="{ row }"><el-tag size="small" type="info">{{ row.label }}</el-tag></template>
      </el-table-column>
      <el-table-column label="时长" width="84">
        <template #default="{ row }">{{ hms(row.duration) }}</template>
      </el-table-column>
      <el-table-column label="说明" min-width="360">
        <template #default="{ row }"><span class="reason">{{ row.reason }}</span></template>
      </el-table-column>
    </el-table>
    <p v-if="showRest" class="hint">
      勾上这里的标题也会导出；它们没有新的文件名，会以原盘里的编号区分——通常不需要。
    </p>
  </div>
</template>

<style scoped>
.titles {
  margin-top: 8px;
}
.streams {
  font-size: 12px;
  color: var(--el-text-color-regular);
  line-height: 1.6;
}
.streams .video {
  font-weight: 600;
}
.dropped {
  text-decoration: line-through;
  color: var(--el-text-color-placeholder);
}
.reason {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.6;
}
.out {
  font-size: 12px;
  word-break: break-all;
}
.rest {
  margin-top: 12px;
}
.hint {
  margin: 6px 0 0;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>

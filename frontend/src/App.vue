<script setup>
import { ref, onMounted } from 'vue'
import { api } from './api'
import HomeView from './views/HomeView.vue'
import SettingsView from './views/SettingsView.vue'
import PromptView from './views/PromptView.vue'

const activeTab = ref('home')
// the app ships as a zip and is updated by replacing files, so seeing which
// build answered is the quickest way to confirm an update actually landed
const version = ref('')

onMounted(async () => {
  try {
    version.value = (await api.getVersion()).version
  } catch {
    version.value = ''  // an old backend has no /api/version; just omit it
  }
})
</script>

<template>
  <el-container class="app">
    <el-header class="header">
      <h1>🎬 MovieTranslator</h1>
      <el-tag v-if="version" size="small" type="info" class="version">v{{ version }}</el-tag>
      <span class="subtitle">电影字幕翻译 · 全局一致性 AI 翻译</span>
    </el-header>
    <el-main>
      <el-tabs v-model="activeTab">
        <el-tab-pane label="翻译任务" name="home">
          <HomeView />
        </el-tab-pane>
        <el-tab-pane label="设置" name="settings">
          <SettingsView />
        </el-tab-pane>
        <el-tab-pane label="提示词" name="prompts">
          <PromptView />
        </el-tab-pane>
      </el-tabs>
    </el-main>
  </el-container>
</template>

<style>
body {
  margin: 0;
}
.app {
  max-width: 1200px;
  margin: 0 auto;
}
.header {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding-top: 16px;
  height: auto !important;
}
.header h1 {
  margin: 0;
  font-size: 24px;
  color: var(--el-text-color-primary);
}
.header .version {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  align-self: center;
}
.header .subtitle {
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
</style>

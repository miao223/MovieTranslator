<script setup>
import { ref } from 'vue'
import HomeView from './views/HomeView.vue'
import SettingsView from './views/SettingsView.vue'
import PromptView from './views/PromptView.vue'

const activeTab = ref('home')
</script>

<template>
  <el-container class="app">
    <el-header class="header">
      <h1>🎬 MovieTranslator</h1>
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
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
  background: #f5f7fa;
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
}
.header .subtitle {
  color: #909399;
  font-size: 14px;
}
</style>

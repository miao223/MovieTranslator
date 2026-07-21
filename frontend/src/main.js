import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import './theme.css'
import App from './App.vue'
import { applyTheme, loadTheme } from './theme'

applyTheme(loadTheme())

createApp(App).use(ElementPlus).mount('#app')

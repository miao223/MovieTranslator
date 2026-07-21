// 界面主题：纯前端偏好，存 localStorage（与后端设置文件无关，换机器不迁移）
export const THEMES = [
  {
    value: 'slate',
    name: '柔雾蓝灰',
    desc: '默认。冷灰底 + 靛蓝主色，柔投影无边框，久看不刺眼。',
    dark: false,
    swatch: ['#E9EDF3', '#F7F9FC', '#4C6FE7'],
  },
  {
    value: 'paper',
    name: '暖白纸感',
    desc: '米黄纸张底 + 墨绿主色，色温偏暖、蓝光最少，适合长时间盯屏。',
    dark: false,
    swatch: ['#F4F1E9', '#FBF9F3', '#4A6B57'],
  },
  {
    value: 'cinema',
    name: '深色影院',
    desc: '深灰底 + 琥珀金主色，暗环境看片时最舒服。',
    dark: true,
    swatch: ['#16181D', '#20242B', '#E8B45E'],
  },
  {
    value: 'terminal',
    name: '终端极客',
    desc: '墨绿黑底 + 荧光绿主色，等宽字体直角边框，工具感最强。',
    dark: true,
    swatch: ['#101614', '#131B18', '#5FD08A'],
  },
]

export const DEFAULT_THEME = 'slate'
const STORAGE_KEY = 'mt-theme'

export function loadTheme() {
  const saved = localStorage.getItem(STORAGE_KEY)
  return THEMES.some((t) => t.value === saved) ? saved : DEFAULT_THEME
}

export function applyTheme(value) {
  const theme = THEMES.find((t) => t.value === value) || THEMES[0]
  const root = document.documentElement
  root.dataset.theme = theme.value
  // Element Plus 暗色组件变量靠 html.dark 生效
  root.classList.toggle('dark', theme.dark)
  localStorage.setItem(STORAGE_KEY, theme.value)
  return theme.value
}

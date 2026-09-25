// The language lists of the translation options, shared by the 翻译任务 page
// and the 原盘 page's 「加入列队并做字幕」 dialog. Both selects that take a
// language name also accept a typed one (filterable allow-create): the
// backend knows more languages than these lists name.

export const SOURCE_LANGS = [
  { value: 'auto', label: '自动检测' },
  { value: 'en', label: '英语' },
  { value: 'ja', label: '日语' },
  { value: 'ko', label: '韩语' },
  { value: 'fr', label: '法语' },
  { value: 'de', label: '德语' },
  { value: 'es', label: '西班牙语' },
  { value: 'ru', label: '俄语' },
  { value: 'zh', label: '中文' },
  { value: 'pt', label: '葡萄牙语' },
  { value: 'it', label: '意大利语' },
  { value: 'th', label: '泰语' },
  { value: 'vi', label: '越南语' },
  { value: 'ar', label: '阿拉伯语' },
  { value: 'hi', label: '印地语' },
]

export const TARGET_LANGS = [
  '简体中文', '繁體中文', 'English', '日本語', '한국어', 'Français', 'Deutsch',
  'Português', 'Italiano', 'ไทย', 'Tiếng Việt', 'العربية', 'हिन्दी',
]

// batch mode picks tracks by language tag: stream indices differ per file
export const AUDIO_LANGS = [
  { value: '', label: '每个文件的默认音轨' },
  { value: 'jpn', label: '日语音轨（jpn）' },
  { value: 'eng', label: '英语音轨（eng）' },
  { value: 'chi', label: '中文音轨（chi/zho）' },
  { value: 'kor', label: '韩语音轨（kor）' },
  { value: 'fre', label: '法语音轨（fre）' },
  { value: 'ger', label: '德语音轨（ger）' },
  { value: 'spa', label: '西班牙语音轨（spa）' },
  { value: 'rus', label: '俄语音轨（rus）' },
  { value: 'por', label: '葡萄牙语音轨（por）' },
  { value: 'ita', label: '意大利语音轨（ita）' },
  { value: 'tha', label: '泰语音轨（tha）' },
  { value: 'vie', label: '越南语音轨（vie）' },
  { value: 'ara', label: '阿拉伯语音轨（ara）' },
  { value: 'hin', label: '印地语音轨（hin）' },
]

// batch mode picks subtitle tracks by language tag, same reason as audio
export const SUB_LANGS = [
  { value: '', label: '每个文件里最合适的一条' },
  { value: 'eng', label: '英语字幕（eng）' },
  { value: 'jpn', label: '日语字幕（jpn）' },
  { value: 'chi', label: '中文字幕（chi/zho）' },
  { value: 'kor', label: '韩语字幕（kor）' },
  { value: 'fre', label: '法语字幕（fre）' },
  { value: 'ger', label: '德语字幕（ger）' },
  { value: 'spa', label: '西班牙语字幕（spa）' },
  { value: 'rus', label: '俄语字幕（rus）' },
  { value: 'por', label: '葡萄牙语字幕（por）' },
  { value: 'ita', label: '意大利语字幕（ita）' },
  { value: 'tha', label: '泰语字幕（tha）' },
  { value: 'vie', label: '越南语字幕（vie）' },
  { value: 'ara', label: '阿拉伯语字幕（ara）' },
  { value: 'hin', label: '印地语字幕（hin）' },
]

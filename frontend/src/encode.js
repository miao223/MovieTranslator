// 压制 (re-encoding), shared by the 压制 page, the settings page and the
// 原盘 page's full-pipeline dialog: what this server can encode with, and
// how an encode is named. The server decides both (services/encode.py);
// this only asks once per page load and spells the answers.
import { api } from './api'

let asked = null

// { encoders: [{id, family, hardware, engine, ten_bit, tunes, quality}],
//   deinterlace } — only encoders this machine can really open
export function encoderCaps() {
  asked ??= api.encoders()
    .then((r) => ({ encoders: r.capabilities || [], deinterlace: !!r.deinterlace }))
    .catch((e) => {
      asked = null   // ask again next time rather than remember a failure
      throw e
    })
  return asked
}

// the formats offered, in the order offered; tag is what an encode written
// next to its source is named with (片名.HEVC.mkv)
export const FAMILIES = [
  { value: 'H.265', tag: 'HEVC', label: 'H.265（HEVC）' },
  { value: 'AV1', tag: 'AV1', label: 'AV1' },
  { value: 'H.264', tag: 'H264', label: 'H.264（AVC）' },
  { value: 'VP9', tag: 'VP9', label: 'VP9' },
]

export const VENDORS = { nvenc: 'NVIDIA（NVENC）', qsv: 'Intel（QSV）', amf: 'AMD（AMF）' }

// what schemas.EncodeOptions defaults to — used only when the settings
// cannot be read; the settings' own `encode` is where the page starts from
export function defaultEncodeOptions() {
  return {
    container: 'mkv', video_codec: 'libx265', rate_control: 'quality', quality: 22,
    bitrate_kbps: 6000, preset: 'medium', tune: '', bit_depth: 'auto', max_height: 0,
    deinterlace: 'auto', audio_codec: 'eac3', audio_scope: 'lossless',
    audio_bitrate_kbps: 0, audio_mixdown: 'keep', audio_languages: [],
    subtitles: 'all', subtitle_languages: [], auto_pick: false,
    ...RESTORE_OFF,
  }
}

// the 修复 fields of EncodeOptions as a plain encode leaves them
export const RESTORE_OFF = {
  field_order: 'auto', crop_top: 0, crop_bottom: 0, crop_left: 0, crop_right: 0,
  crop_auto: false, aspect: 'auto', denoise: 'off', upscale: 0, ai_model: '',
}

// the deinterlace mode an encode will really use: 自动判断 is decided by
// what the picture is (GET /api/restore/analyze), when that is known
export function effectiveDeinterlace(options, analysis) {
  if (options.deinterlace !== 'detect') return options.deinterlace
  return analysis?.cadence?.mode || ''
}

// 片名.1080p.60fps.HEVC.mkv: what a restore did to the picture, as the server
// names it (restore.name_tags). *fps* is the source's, from its probe.
export function restoreTags(options, fps, analysis = null) {
  const tags = []
  if (options.upscale) tags.push(`${options.upscale}p`)
  if (effectiveDeinterlace(options, analysis) === 'bob' && fps && fps <= 31) {
    tags.push(`${Math.round(fps * 2)}fps`)
  }
  return tags
}

export function familyOf(encoders, codec) {
  if (codec === 'copy') return 'copy'
  return encoders.find((e) => e.id === codec)?.family || ''
}

// 片名.HEVC.mkv next to the source, 片名.mkv in a chosen folder
export function encodedName(stem, encoders, options, custom, fps = 0, analysis = null) {
  const ext = options.container || 'mkv'
  if (custom) return `${stem}.${ext}`
  const family = familyOf(encoders, options.video_codec)
  const tag = family === 'copy' ? 'remux'
    : FAMILIES.find((f) => f.value === family)?.tag || options.video_codec
  return [stem, ...restoreTags(options, fps, analysis), tag, ext].join('.')
}

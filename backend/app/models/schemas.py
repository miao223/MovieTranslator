"""Pydantic models shared by settings, jobs and the API layer."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- settings


class LLMSettings(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    # vision model for on-screen text translation and graphic-subtitle OCR;
    # empty = use `model` (needed because strong text models like DeepSeek
    # have no vision)
    vision_model: str = ""
    # ...and its own endpoint, because the vision model is usually somewhere
    # else entirely: a VL model running on the local GPU while translation
    # goes to a cloud provider. Empty = the endpoint above. Both stages run
    # inside one job, so without this only one of them can be reached.
    vision_base_url: str = ""
    vision_api_key: str = ""
    # ...and the same again for the model that listens, used when
    # ASRSettings.engine is "api". Its own endpoint for the same reason as
    # the vision one: recognition runs at the start of a job and translation
    # at the end, so both have to be reachable at once. Empty = the main
    # endpoint; the key is never inherited across a different base_url.
    audio_model: str = ""
    audio_base_url: str = ""
    audio_api_key: str = ""
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    # Lines translated per output batch, bounded by the model's max output
    # tokens. Bigger batches mean fewer round trips and a conversation that
    # grows more slowly, so a film costs less time and fewer tokens. The
    # reason to keep them small used to be that a long batch gives the model
    # more room to lose its place and shift every line after it — now that
    # each request restates its own source lines and every batch is checked
    # for alignment (see translator.py), that risk is covered, and the
    # check's own margin is wider at 200 than at 120 on real runs.
    batch_size: int = Field(200, ge=1, le=500)
    # Ask the provider to skip its reasoning pass. Both jobs here are
    # mechanical, and DeepSeek documents that thinking mode also disables
    # temperature — so the temperature=0 the preprocessing pass requests
    # only takes effect once this is on. Providers that do not know the
    # parameter are detected and stop being sent it.
    disable_thinking: bool = True
    # approximate context window of the model, in tokens; beyond this the
    # translator falls back to sliding-window chunking
    context_limit: int = Field(100_000, ge=1_000)


class NetworkSettings(BaseModel):
    """Outbound proxy. Inbound binding lives in ServerSettings."""

    proxy_url: str = ""  # e.g. http://127.0.0.1:7890
    llm_via_proxy: bool = False
    model_download_via_proxy: bool = False


class ServerSettings(BaseModel):
    """How the app listens. Off by default: loopback only, as before.

    Every endpoint here assumes the caller is the person sitting at the
    machine — /api/fs/browse lists any directory, /api/settings returns the
    LLM key, /api/jobs starts work on any path. So binding beyond loopback
    and requiring a token are one feature, not two.
    """

    # bind 0.0.0.0 instead of 127.0.0.1; takes effect on restart
    lan_access: bool = False
    port: int = Field(8760, ge=1, le=65535)
    # whether non-loopback requests need the token. Loopback is ALWAYS
    # exempt, so a forgotten token can never lock the local user out.
    require_token: bool = True
    # generated when LAN access is switched on; never written to any log
    access_token: str = ""


class MCPSettings(BaseModel):
    """MCP server (services/mcp_server.py), mounted at /mcp.

    Off by default; while off the mount answers 404. Independent of
    lan_access — it can serve a client on this same machine.
    """

    enabled: bool = False


class ASRSettings(BaseModel):
    # non-empty: use this local CTranslate2 model directory, ignore model_size
    model_path: str = ""
    model_size: str = "large-v2"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = Field(5, ge=1, le=10)
    # per-word timestamps: real time boundaries for line segmentation,
    # eliminates stretched/fabricated cue timings (~10-20% slower)
    word_timestamps: bool = True
    vad_filter: bool = True
    # silero-VAD tuning; threshold defaults below faster-whisper's 0.5
    # because movie dialog is often quiet and gets skipped at 0.5
    vad_threshold: float = Field(0.35, ge=0.05, le=0.95)
    # short interjections ("えっ", "うん", "喂") are real subtitle lines and
    # were being dropped wholesale at 250ms
    vad_min_speech_ms: int = Field(100, ge=0, le=5000)
    vad_min_silence_ms: int = Field(2000, ge=100, le=10000)
    vad_speech_pad_ms: int = Field(400, ge=0, le=3000)
    # Second ASR pass over the stretches the VAD rejected. Silero misses
    # speech that sits under music — one film's narration transcribed fine
    # while every dramatised scene came back empty, and re-running just
    # those stretches with the VAD off recovered 35 lines where the normal
    # pass produced none. Costs roughly a second transcription in time, and
    # more LLM tokens downstream because there is more text to process.
    second_pass: bool = True
    # Fallback for sources silero cannot hear at all: run the FIRST pass as
    # VAD-off windows over the whole timeline, instead of letting the VAD
    # gate it and leaving the second pass to recover the rest. Off by
    # default and self-limiting — it only engages when the VAD kept under
    # 10% of the film AND there is at least that much audio outside it at
    # speech loudness (asr.windowed_first_pass). Measured: a VHS capture
    # kept 5.2% and had 5366s of loud audio outside, while a DVD kept 28%
    # and a Blu-ray 64%, so neither of those can trigger it.
    windowed_first_pass: bool = False
    # source-language hint fed to whisper as `initial_prompt`: proper nouns
    # it keeps mis-hearing (character names above all). MUST be written in
    # the spoken language — a prompt in another language drags the whole
    # transcript toward that language, which is why the plot synopsis and
    # the translation glossary are deliberately NOT reused here.
    initial_prompt: str = ""
    # Prepend a punctuated, capitalised sample sentence in the source
    # language to `initial_prompt`. Whisper conditions its output on that
    # prompt, so the prompt's shape is the shape it tends to write in — and
    # a transcript with no sentence-final punctuation is what pushes the
    # segmenter onto its "merge nothing here, let refine do it" path (two
    # Japanese films came back 72% and 57% open-ended). Off by default: it
    # changes decoding, and a change to decoding has to be measured before
    # it becomes the default (asr.STYLE_EXEMPLARS).
    style_prompt: bool = False
    # Local faster-whisper, or a multimodal LLM that takes audio (see
    # services/asr_api.py, credentials in LLMSettings.audio_*). Local is the
    # default for the same reason OcrSettings.engine's is: it costs nothing,
    # needs no network, and a switch that starts uploading the film's
    # soundtrack on its own is the wrong default whatever its quality.
    engine: Literal["local", "api"] = "local"
    # api engine: seconds of audio per request. Measured on gemini-3.8-flash
    # through a relay: 300s of audio costs ~7.5k input tokens and drifts by
    # ≤±0.5s inside the window. The cap is not a cost limit but a trust one —
    # a model whose clock slips does it progressively, so a window that runs
    # for ten minutes makes its own timestamps unusable. Each window is
    # anchored to a locally known offset, so drift can never accumulate
    # across a film.
    api_window_seconds: float = Field(300.0, ge=60.0, le=420.0)
    # mp3 is ~1/8 the bytes of wav at 16kHz mono and both OpenAI and Gemini
    # accept it; wav is the escape hatch for an endpoint that will not
    # decode mp3.
    api_audio_format: Literal["mp3", "wav"] = "mp3"
    # windows in flight at once. Unlike whisper this engine is network-bound,
    # so a film is as fast as the endpoint allows; raise it only as far as
    # the endpoint's rate limit.
    api_concurrency: int = Field(3, ge=1, le=8)


class SubtitleSettings(BaseModel):
    max_chars_per_line: int = Field(42, ge=10, le=120)
    max_duration: float = Field(6.0, ge=1.0, le=15.0)
    # where the translated line sits in a bilingual cue
    bilingual_layout: Literal["translation_top", "translation_bottom"] = (
        "translation_bottom"
    )
    # styled output: emits .ass instead of .srt (SRT cannot carry font
    # size/color reliably); sizes are for a 1920x1080 canvas
    style_enabled: bool = False
    font_size: int = Field(56, ge=16, le=120)  # translation line
    original_font_size: int = Field(40, ge=12, le=120)
    translation_color: str = "#FFFFFF"
    original_color: str = "#B4B4B4"


DEFAULT_TONE = "语言口语化、符合角色语气，适合字幕阅读，简洁不啰嗦。"


class PromptSettings(BaseModel):
    # transcript preprocessing pass (services/refine.py): rejoins sentences
    # the ASR cut apart before translation, so no line holds a meaningless
    # fragment the translator would fill with the next line's content
    refine_enabled: bool = True
    # switches targeting known ASR weaknesses
    fix_asr_errors: bool = True      # homophone / mis-recognition correction
    link_fragments: bool = True      # cross-line coherence for fragmented lines
    normalize_loanwords: bool = True # katakana / transliterated loanword handling
    limit_length: bool = True        # keep translation subtitle-length friendly
    # Find what is sung rather than spoken and wrap it in ♪ (services/lyrics.py).
    # On by default: without it a song's fate depends only on which stage it
    # landed in — first-pass lyrics get translated as dialogue, second-pass
    # ones get thrown away — and that is incoherent whether or not the film
    # has songs. Costs one extra pass over the transcript (~+22% tokens on a
    # song-heavy film). While off, lyrics the first pass picked up stay in the
    # subtitle as ordinary dialogue — only the second pass's are dropped,
    # because deleting first-pass content is a power vetting deliberately does
    # not have.
    mark_lyrics: bool = True
    # style requirements (rule 3 of the system prompt)
    tone: str = DEFAULT_TONE
    # user-provided glossary, one "原文 → 译文" per line; always obeyed
    glossary: str = ""
    # free-form extra instructions appended to the system prompt
    extra: str = ""
    # advanced: full override; supports {target_language} / {synopsis} placeholders
    custom_system_prompt: str = ""


class OcrSettings(BaseModel):
    """Reading graphic subtitle tracks (PGS/VobSub/DVB) — services/ocr.py."""

    # rapidocr runs locally and free; vision goes through llm.vision_model.
    # The local one is the default because it costs nothing and needs no
    # network, and because a switch that starts spending money on its own
    # is the wrong default whatever its quality.
    engine: Literal["rapidocr", "vision"] = "rapidocr"
    # recognition language, e.g. "ja" / "en"; empty = the track's own tag,
    # and failing that a probe over the first few cues
    language: str = ""
    # recognisers want tall text; DVD subtitles at 720x480 need the help
    upscale: int = Field(2, ge=1, le=4)
    # vision engine: cues per request. Bigger is cheaper and faster, but a
    # sheet the model loses count on is discarded whole.
    vision_batch: int = Field(10, ge=1, le=40)
    # There is no separate proofreading switch: OCR output goes through the
    # same 转写预处理 pass as speech recognition (prompts.refine_enabled),
    # with wording aimed at look-alike glyphs instead of homophones.


class DiscSettings(BaseModel):
    """Remuxing Blu-ray / DVD / ISO discs to MKV (services/disc)."""

    # Extras (behind-the-scenes, trailers, creditless openings…) are ticked
    # by default: the user's call when the feature was designed. Each disc
    # can still be changed title by title before it is queued.
    export_extras: bool = True
    # Titles shorter than this are logos, warnings and menu backgrounds —
    # listed, never ticked by default.
    min_title_seconds: int = Field(60, ge=0, le=3600)
    # Matroska cannot hold a disc's own LPCM formats (pcm_bluray / pcm_dvd),
    # so those tracks are converted — losslessly either way. FLAC is about
    # half the size; PCM is for players that cannot decode FLAC.
    lpcm: Literal["flac", "pcm"] = "flac"
    # Where the MKVs go unless the 原盘 page says otherwise (DiscOutputMode).
    # "beside" by the user's choice: Film (1992)/ → Film (1992).mkv next to
    # it, so a library folder ends up holding films, not disc folders to
    # open; a box set's volumes then land together in the set's folder.
    output_mode: Literal["beside", "inside", "custom"] = "beside"
    output_dir: str = ""             # for "custom"


class EncodeOptions(BaseModel):
    """What a 压制 (re-encode) job does to a video (services/encode.py).

    One model for three places: the 压制 page, the defaults on the settings
    page (AppSettings.encode) and the full-pipeline dialog on the 原盘 page.
    The defaults are the user's choice: H.265 10bit, the lossless audio
    tracks to E-AC-3, everything else kept as it is.
    """

    container: Literal["mkv", "mp4"] = "mkv"
    # encoder id (libx265, hevc_nvenc…), or "copy" to leave the picture
    # alone. A free string for the reason EmbedSettings.video_codec is one:
    # which encoders exist depends on the machine, and the server validates
    # against its own probe (encode.capabilities).
    video_codec: str = "libx265"
    rate_control: Literal["quality", "bitrate"] = "quality"
    # CRF-style, lower is better. The scale is the encoder's own — x26x and
    # NVENC 0-51, SVT-AV1 / VP9 0-63, AMF's AV1 0-255 — and out-of-range
    # values are clamped to it (encode.capabilities says which).
    quality: int = Field(22, ge=0, le=255)
    bitrate_kbps: int = Field(6000, ge=100, le=200_000)  # rate_control="bitrate"
    preset: Literal["ultrafast", "fast", "medium", "slow", "veryslow"] = "medium"
    # content tuning. Only x264 (film/animation/grain) and x265 (animation/
    # grain — it refuses "film" at open) have it; ignored elsewhere.
    tune: Literal["", "film", "animation", "grain"] = ""
    # auto: H.265 / AV1 in 10bit, H.264 / VP9 in 8bit (H.264 High 10 barely
    # plays on hardware decoders). An HDR source is always 10bit.
    bit_depth: Literal["auto", "8", "10"] = "auto"
    # a limit on the shorter side; 0 keeps the size. Never upscales.
    max_height: Literal[0, 2160, 1440, 1080, 720, 576, 480] = 0
    # auto: only frames the source flags as interlaced (bwdif); all: every
    # frame; ivtc: undo 3:2 pulldown (NTSC film on DVD) back to 23.976
    deinterlace: Literal["auto", "off", "all", "ivtc"] = "auto"
    audio_codec: Literal["copy", "aac", "libopus", "ac3", "eac3", "flac"] = "eac3"
    # which tracks audio_codec applies to. "lossless": TrueHD, DTS-HD MA,
    # PCM, FLAC… only — re-encoding an AC-3 or DTS core into another lossy
    # format loses quality for little space, so those are copied.
    audio_scope: Literal["lossless", "all"] = "lossless"
    audio_bitrate_kbps: int = Field(0, ge=0, le=6144)   # 0 = by channel count
    audio_mixdown: Literal["keep", "stereo"] = "keep"
    # ISO 639-2 tags of the audio tracks to keep; empty keeps every one.
    # Untagged tracks always stay: there is nothing to judge them by.
    audio_languages: list[str] = []
    subtitles: Literal["all", "languages", "none"] = "all"
    subtitle_languages: list[str] = []
    # 按画面自动选编码 (services/encodepick.py): before the encode starts a
    # vision model looks at a contact sheet of the film and says what it is
    # — animation or live action, how grainy — and video_codec, preset,
    # quality, tune and rate_control are taken from the measured table
    # instead of from here. If it cannot say, the fields above are used.
    auto_pick: bool = False


class AppSettings(BaseModel):
    # temp working dir for intermediate files; empty = platform cache dir.
    # only its "jobs" subdirectory is managed (and wiped on startup)
    work_dir: str = ""
    # where downloaded whisper models are stored; empty = HuggingFace default
    # cache (~/.cache/huggingface/hub). changing it does NOT move old models
    model_cache_dir: str = ""
    # Half-finished work kept so an interrupted film can carry on instead of
    # buying its audio and transcription again (core/cache.py). Two limits,
    # whichever bites first; audio is ~230 MB for a two-hour film, so the
    # size one usually does. Either at 0 turns resuming off and keeps
    # nothing — the setting for someone who would rather spend tokens than
    # disk.
    checkpoint_days: int = Field(3, ge=0, le=90)
    checkpoint_max_gb: int = Field(5, ge=0, le=500)
    # deep-diagnostics log next to the subtitle output (core/debuglog.py):
    # raw ASR output, every segmentation/merge decision, full LLM traffic.
    # off by default because the file runs to several MB per film
    debug_mode: bool = False
    llm: LLMSettings = LLMSettings()
    asr: ASRSettings = ASRSettings()
    subtitle: SubtitleSettings = SubtitleSettings()
    ocr: OcrSettings = OcrSettings()
    prompts: PromptSettings = PromptSettings()
    network: NetworkSettings = NetworkSettings()
    server: ServerSettings = ServerSettings()
    mcp: MCPSettings = MCPSettings()
    disc: DiscSettings = DiscSettings()
    # what the 压制 page and the 原盘 page's full pipeline start from
    encode: EncodeOptions = EncodeOptions()


# ---------------------------------------------------------------- jobs


class FrameTask(BaseModel):
    """One on-screen-text translation point (画面翻译)."""

    time: str  # "1:23:45" / "23:45" / "85"
    note: str = ""  # hint for the vision model, e.g. "手机短信内容"
    duration: float = Field(5.0, ge=1.0, le=60.0)  # cue display seconds


class EmbedSettings(BaseModel):
    """Container and codec choices for the muxed video (services/mux.py).

    Only meaningful when JobRequest.embed_subtitle is on. The defaults are
    exactly what the feature did before these existed: an mkv with every
    stream copied, so nothing re-encodes unless it was asked for.
    """

    container: Literal["mkv", "mp4"] = "mkv"
    # encoder id, or "copy" to remux the picture untouched. Deliberately a
    # free string rather than a Literal: which encoders exist depends on the
    # machine (NVENC/QSV/AMF need the hardware), so the list comes from
    # GET /api/media/encoders and the server validates against that same
    # probe. ASRSettings.model_size / compute_type are free strings for the
    # same reason.
    video_codec: str = "copy"
    audio_codec: Literal[
        "copy", "aac", "ac3", "eac3", "flac", "libopus"
    ] = "copy"
    # CRF-style quality, lower is better. The scales differ per encoder
    # (x264/x265 0-51, SVT-AV1 0-63, NVENC calls it cq); mux.py maps and
    # clamps this into whatever the chosen encoder actually wants.
    quality: int = Field(23, ge=0, le=63)
    # abstract speed tier, mapped per encoder — x26x take these names as-is,
    # SVT-AV1 wants a number, NVENC wants p1-p7
    preset: Literal[
        "ultrafast", "fast", "medium", "slow", "veryslow"
    ] = "medium"


class AudioTrack(BaseModel):
    """One audio stream of a video, as offered for selection."""

    index: int  # container stream index, the value passed back as audio_track
    codec: str = ""
    language: str = ""  # canonical ISO 639-2/B code, "" when untagged
    language_name: str = ""
    title: str = ""
    channels: int = 0
    channel_name: str = ""
    sample_rate: int = 0
    default: bool = False


class SubtitleTrack(BaseModel):
    """One subtitle already in (or beside) the video, offered as the source.

    Reading these instead of transcribing is both more accurate and far
    faster — see services/subsource.py.
    """

    index: int  # container stream index; -1 for a sidecar file
    path: str = ""  # sidecar file path; empty for an embedded track
    codec: str = ""
    language: str = ""  # canonical ISO 639-2/B code, "" when untagged
    language_name: str = ""
    title: str = ""
    default: bool = False
    forced: bool = False  # signs only — never auto-selected
    # False for PGS/VobSub/DVB: pictures, with no text to read
    text: bool = True


class JobRequest(BaseModel):
    video_path: str
    # where the original text comes from: speech recognition, or a subtitle
    # the release already carries (services/subsource.py)
    text_source: Literal["asr", "subtitle"] = "asr"
    # container stream index of the subtitle track to read; None = pick one
    subtitle_track: Optional[int] = None
    # a sidecar subtitle file to read instead of an embedded track
    subtitle_file: str = ""
    # fallback when subtitle_track is None (batch jobs, where indices differ
    # per file): prefer a track tagged with this language, e.g. "eng"
    subtitle_language: str = ""
    # Set by BatchManager, never by a single-file request: transcribe when
    # this file has no readable subtitle. A directory of episodes should not
    # stop because one of them lacks a track, while someone who picked a
    # track by hand for one film expects a clear failure, not a silent hour
    # of recognition.
    subtitle_fallback_asr: bool = False
    # container stream index of the audio track to transcribe; None = the
    # track flagged default, else the first one
    audio_track: Optional[int] = None
    # fallback when audio_track is None (batch jobs, where indices differ per
    # file): prefer a track tagged with this language, e.g. "jpn"
    audio_language: str = ""
    source_language: str = "auto"  # whisper language code or "auto"
    target_language: str = "简体中文"
    synopsis: str = ""  # optional plot synopsis to steer the translation
    # bilingual 双语；translation_only 只要译文；original_only 只要原文——
    # 跳过 LLM 翻译，但转写预处理 / 歌词识别 / 二次识别复核 / 图形字幕 OCR
    # 与校对一个都不少。target_language 在这个模式下只作用于画面翻译。
    # bilingual_split 内容与 bilingual 完全相同，只是写成两个单语字幕文件
    # （片名.zh.srt + 片名.en.srt），播放器里当成两条可选字幕。
    # 四种模式的产物都带语言后缀，后缀就是文件里的内容：双语两个都写、谁在
    # 上面谁在前（片名.en-zh.srt），纯译文写目标语言，纯原文写片子自己的
    # 语言。见 pipeline._naming。
    output_mode: Literal[
        "bilingual", "translation_only", "original_only", "bilingual_split"
    ] = "bilingual"
    # produce one new .mkv carrying the subtitle as a switchable soft track
    # instead of a subtitle file next to the video (services/mux.py). The
    # picture is copied, never re-encoded.
    embed_subtitle: bool = False
    # container/codec for that new video. Ignored when embed_subtitle is off.
    embed: EmbedSettings = EmbedSettings()
    # set by BatchManager in series mode; names the glossary this job shares
    # with the rest of its batch (services/series.py). Always empty for a
    # single-file job — nothing it decides can leak into another film.
    series_id: str = ""
    frame_tasks: list[FrameTask] = []
    # supplement mode: skip ASR/translation entirely, only translate the
    # frame_tasks and merge them into the existing same-stem .srt/.ass
    frame_only: bool = False


class BatchRequest(BaseModel):
    directory: str
    recursive: bool = True
    skip_existing_srt: bool = True
    # per-file task params, shared by every video in the batch
    # track indices differ per file, so batches select by language tag instead
    audio_language: str = ""  # e.g. "jpn"; empty = each file's default track
    # see JobRequest.text_source. A file with no readable subtitle falls back
    # to speech recognition rather than failing the batch.
    text_source: Literal["asr", "subtitle"] = "asr"
    subtitle_language: str = ""  # e.g. "eng"; empty = each file's best track
    source_language: str = "auto"
    target_language: str = "简体中文"
    synopsis: str = ""  # shared synopsis is useful for TV series batches
    # see JobRequest.output_mode — original_only 跳过翻译，只输出原文；
    # bilingual_split 每个视频产出译文 + 原文两个文件
    output_mode: Literal[
        "bilingual", "translation_only", "original_only", "bilingual_split"
    ] = "bilingual"
    # see JobRequest.embed_subtitle — one muxed .mkv per video instead of a
    # subtitle file. Note it writes a second copy of every film in the batch.
    embed_subtitle: bool = False
    # see JobRequest.embed — same container/codec choice for every file
    embed: EmbedSettings = EmbedSettings()
    # 剧集模式: every video in this batch shares one accumulated 原文 → 译名
    # table, so a name settled in one episode holds for the rest
    # (services/series.py). Off by default — a directory of unrelated films
    # would only contaminate each other's names.
    series_mode: bool = False


class DiscSubtitles(BaseModel):
    """原盘页「加入列队并做字幕」: what every MKV of a remux is translated
    with, as confirmed in the button's dialog. The fields a translation
    batch shares (BatchRequest) minus the folder, with the same defaults —
    the app keeps no saved defaults of its own for these, on purpose."""

    text_source: Literal["asr", "subtitle"] = "asr"
    audio_language: str = ""         # "" = each file's default track
    subtitle_language: str = ""      # "" = each file's best track
    source_language: str = "auto"
    target_language: str = "简体中文"
    output_mode: Literal[
        "bilingual", "translation_only", "original_only", "bilingual_split"
    ] = "bilingual"
    # a muxed MKV is a second copy of every file: the page says so
    embed_subtitle: bool = False
    # one 原文 → 译名 table per disc, or per box set (services/series.py) —
    # never per batch, where unrelated films would share their names
    series_mode: bool = True


DiscOutputMode = Literal["beside", "inside", "custom"]


class DiscRequest(BaseModel):
    """Remux titles of one disc to MKV (原盘封装). Queue-only: a disc is
    added to the queue from its own page, never started directly."""

    path: str                        # disc folder, BDMV/VIDEO_TS, a file inside, or .iso
    # title ids as GET /api/disc/scan reported them ("00800.mpls", "title03",
    # "00001.mpls#2"). Empty = whatever the analysis ticks by default.
    titles: list[str] = []
    # base name of the outputs; "" = the disc's own, or its box set's
    name: str = ""
    # "beside" = in the folder the disc is in (Film/ → Film.mkv next to it);
    # "inside" = in the disc's own folder (an .iso gets one, named after
    # it); "custom" = output_dir. None = DiscSettings.output_mode. An entry
    # queued before the modes existed has none and a non-empty output_dir
    # only when one was chosen — read as "custom" (pipeline._execute_disc).
    output_mode: Optional[DiscOutputMode] = None
    output_dir: str = ""
    # None = automatic: 1, or — for a volume of a box set — on from the
    # volumes before it. Resolved when the entry is queued, like `titles`:
    # the run must not depend on the discs next to this one (VOL01 may be
    # gone by then).
    episode_start: Optional[int] = Field(None, ge=0, le=9999)
    # set when queued, for a volume of a box set: its extras number on from
    # the other volumes', and what is numbered per disc is named with
    # own_name (analyze.analyze)
    extra_start: Optional[int] = Field(None, ge=1, le=9999)
    own_name: str = ""
    # the 整片/分集 answer; None = let the disc's structure decide. The same
    # value must reach the run as reached the scan, or ids like
    # "00001.mpls#2" would not exist when the work starts.
    series: Optional[bool] = None

    def effective_output_mode(self) -> str:
        """output_mode, for an entry queued before the modes too: it had
        only output_dir — a folder when one was chosen, empty for the
        default place, which is "beside" now."""
        return self.output_mode or ("custom" if self.output_dir.strip() else "beside")


class DiscEnqueueRequest(DiscRequest):
    """POST /api/queue/disc: a DiscRequest, and optionally what to translate
    its MKVs with once they are written. Kept off DiscRequest itself: that
    is the remux job's own payload (its JSON is in the checkpoint key), and
    the subtitles are the queue's business, not the remux's."""

    subtitles: Optional[DiscSubtitles] = None
    # 加入列队并压制、做字幕: encode every MKV first (DiscEncode)
    encode: Optional["DiscEncode"] = None


class DiscAnswer(BaseModel):
    """What the 原盘 page's batch mode was told about one disc of the folder."""

    path: str                        # DiscReport.path, as the scan gave it
    series: Optional[bool] = None
    name: str = ""
    episode_start: Optional[int] = Field(None, ge=0, le=9999)
    titles: list[str] = []           # for 加入列队: the ticks; [] = the defaults
    include: bool = True             # for 加入列队: this disc is wanted at all


class DiscBatchRequest(BaseModel):
    """The batch mode's scan and its 加入列队: every disc in a folder."""

    path: str                        # the folder
    recursive: bool = True
    output_mode: Optional[DiscOutputMode] = None
    output_dir: str = ""
    # answers given so far, by disc; a disc without one gets the defaults.
    # 加入列队 queues exactly the discs listed here with include=True — not
    # whatever the folder holds by then.
    discs: list[DiscAnswer] = []
    # 加入列队并做字幕 (ignored by the scan): see DiscEnqueueRequest
    subtitles: Optional[DiscSubtitles] = None
    encode: Optional["DiscEncode"] = None


class DiscEncode(BaseModel):
    """原盘页「加入列队并压制、做字幕」: what the disc's MKVs are encoded with
    once they are remuxed, before their subtitles are made."""

    options: EncodeOptions = EncodeOptions()
    # The user's default: the lossless MKV goes once its encode has been
    # verified, and the encode takes its name. The disc itself is never
    # touched, so the lossless one can always be remuxed again.
    keep_lossless: bool = False


class EncodeRequest(BaseModel):
    """Re-encode one video (压制). Queue-only, like a disc."""

    source: str
    options: EncodeOptions = EncodeOptions()
    # "beside": next to the source, the codec in the name (片名.HEVC.mkv);
    # "custom": in output_dir under the source's own name
    output_mode: Literal["beside", "custom"] = "beside"
    # for "custom": this file's folder — a batch keeps each file's
    # sub-folder under the chosen one, worked out when it is queued
    output_dir: str = ""
    # Swap the source for the encode once it is verified. Set by the queue
    # alone, from a disc's then_encode: the source is then the lossless MKV
    # this program remuxed moments before, and it carries the tag that says
    # so. The queue endpoints refuse it — the 压制 page never deletes or
    # overwrites anything of the user's.
    replace_source: bool = False


class EncodeEnqueueRequest(BaseModel):
    """POST /api/queue/encode: one file, and optionally its subtitles after."""

    source: str
    options: EncodeOptions = EncodeOptions()
    output_mode: Literal["beside", "custom"] = "beside"
    output_dir: str = ""
    subtitles: Optional[DiscSubtitles] = None


class EncodeBatchRequest(BaseModel):
    """POST /api/queue/encode-batch: the files the 压制 page's batch mode
    ticked, from one scanned folder."""

    path: str                        # the folder the files were found in
    files: list[str] = []
    options: EncodeOptions = EncodeOptions()
    output_mode: Literal["beside", "custom"] = "beside"
    output_dir: str = ""
    subtitles: Optional[DiscSubtitles] = None


class EncodeScanRequest(BaseModel):
    path: str
    recursive: bool = True


class DiscStreamInfo(BaseModel):
    kind: str
    codec: str
    language: str = ""
    language_name: str = ""
    detail: str = ""
    forced: bool = False
    commentary: bool = False
    carried: bool = True             # False: listed on the disc, not in the MKV
    note: str = ""


class DiscTitleInfo(BaseModel):
    id: str
    number: int
    label: str = ""                  # 正片 / 第 3 集 / 花絮 02 / 重复 …
    category: str                    # main episode extra variant duplicate short loop still silent
    selected: bool = False           # the default tick
    selectable: bool = True
    reason: str = ""
    duplicate_of: str = ""
    output: str = ""                 # file name it would get (before any .2)
    ordinal: int = 0                 # episode / extra / version number; 0 for the rest
    duration: float = 0.0
    chapters: int = 0
    size: Optional[int] = None       # bytes; None when unknown
    video: str = ""
    streams: list[DiscStreamInfo] = []
    angles: int = 1
    segments: int = 0
    reachable: bool = False
    notes: list[str] = []


class DiscVolumeInfo(BaseModel):
    """This disc as one volume of a box set (services/disc/volumes.py)."""

    id: str                          # the same for every volume of the set
    name: str                        # the set's shared name
    index: int                       # 1-based
    count: int
    members: list[str] = []          # the volumes' folder / image names, in order
    # False: a volume that is not a series volume (a bonus disc judged a
    # film) — it keeps its own name and numbering
    chained: bool = True
    note: str = ""


class DiscReport(BaseModel):
    path: str
    root: str
    kind: Literal["bd", "dvd"]
    source: Literal["dir", "iso"]
    name: str                        # the base name in effect
    label: str = ""                  # the disc's own title, when it has one
    output_mode: DiscOutputMode = "beside"
    output_dir: str                  # resolved; "" = a custom folder not chosen yet
    mode: Literal["movie", "series"]
    mode_reason: str = ""
    series_choice: bool = False      # show the 整片/分集 switch
    series_default: bool = False     # what the disc's structure alone says
    series: Optional[bool] = None    # the answer this report was built with
    episode_start: int = 1           # in effect (automatic or the user's)
    extra_start: int = 1
    volume: Optional[DiscVolumeInfo] = None
    titles: list[DiscTitleInfo] = []
    analysis_only: bool = False      # metadata only: can be analysed, not remuxed
    encrypted: bool = False
    free_bytes: Optional[int] = None
    warnings: list[str] = []


class DiscSkipped(BaseModel):
    path: str
    reason: str


class DiscBatchReport(BaseModel):
    path: str
    recursive: bool = True
    output_mode: DiscOutputMode = "beside"
    output_dir: str = ""
    discs: list[DiscReport] = []
    skipped: list[DiscSkipped] = []  # found but unreadable, with why


JobStage = Literal[
    "pending",
    # remuxing a disc's titles to MKV (DiscRequest) — a job of its own kind
    "remuxing",
    # re-encoding a video (EncodeRequest) — likewise
    "encoding",
    "extracting",
    # reading a subtitle the release already carries, in place of
    # extracting + transcribing (text_source="subtitle")
    "importing",
    "transcribing",
    "refining",
    "translating",
    "composing",
    "done",
    "failed",
    "cancelled",
]


class SubtitleLine(BaseModel):
    index: int  # 1-based line number, the key used with the LLM
    start: float  # seconds
    end: float
    text: str
    translation: str = ""
    # on-screen text cue (画面翻译): rendered top-left, translation only
    is_frame: bool = False
    # sung, not spoken (services/lyrics.py) — displayed wrapped in ♪
    is_lyric: bool = False


class JobStatus(BaseModel):
    id: str
    stage: JobStage = "pending"
    progress: float = 0.0  # 0..100 overall
    message: str = ""
    error: Optional[str] = None
    video_path: str = ""
    srt_filename: str = ""  # full path of the generated SRT
    srt_in_place: bool = False  # True when saved next to the video
    # bilingual_split 的另一半（原文那一份）。其余模式恒为空串，所以界面只靠
    # 这一个字段就能决定要不要显示第二个下载按钮。片源目录里已有同名文件时，
    # 产物会带编号让路（片名.zh.2.srt）——本程序绝不覆盖用户的文件。
    original_srt_filename: str = ""
    # embed mode: the new video carrying the subtitle track. Empty otherwise,
    # so the UI can tell the two outcomes apart from this field alone.
    video_filename: str = ""
    # disc remux: every MKV written; 压制: the encode. Full paths, empty for
    # every other job.
    outputs: list[str] = []


class ProgressEvent(BaseModel):
    stage: JobStage
    progress: float
    message: str = ""
    log: str = ""


class BatchStatus(BaseModel):
    id: str
    directory: str
    total: int
    pending: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    current_job_id: str = ""  # the job currently executing, for SSE attach
    jobs: list[JobStatus] = []
    skipped: list[str] = []
    # series mode: the 原文 → 译名 table accumulated so far, one per line.
    # Empty when the mode is off, so the UI can key off it directly.
    glossary: str = ""


# ------------------------------------------------------------------ queue


class QueueEntry(BaseModel):
    """One persisted item of work, with the settings it was enqueued under.

    The snapshot is the whole point: a job keeps the settings that were in
    effect when the button was pressed, so editing settings afterwards
    steers the next job rather than the ones already lined up.

    `settings` is None only when a snapshot written by another version
    could not be validated even leniently; the entry then falls back to
    live settings and says so in `note`.
    """

    id: str
    kind: str = "job"          # a free string, not a Literal: an entry from a
                               # newer version should report itself, not fail.
                               # "job" = translate (request), "disc" = remux
                               # (disc), "encode" = re-encode (encode)
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    title: str = ""            # survives even when the rest cannot be parsed
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    job_id: str = ""
    request: Optional[JobRequest] = None
    disc: Optional[DiscRequest] = None
    encode: Optional[EncodeRequest] = None
    settings: Optional[AppSettings] = None
    error: str = ""
    note: str = ""
    # How many times this entry was interrupted by the process dying. Kept
    # separate from `note`, which is cleared when the entry starts again:
    # "it ran" and "it had to be started over twice" are different facts,
    # and the second one is the only sign that something keeps killing the
    # server. Never reset.
    interrupted: int = 0
    # Copied off JobStatus when the job ends. A restart empties
    # JobManager.jobs, so without these a finished entry could not say what
    # it produced — and the queue exists to survive restarts.
    result_srt: str = ""
    result_srt_original: str = ""   # bilingual_split 的原文那一份
    result_video: str = ""
    result_in_place: bool = False
    # disc remux: every MKV it wrote; 压制: the encode (JobStatus.outputs)
    result_files: list[str] = []
    # Files enqueued from one directory share these, so the UI can collapse
    # them into a single row instead of drowning the list.
    group_id: str = ""
    group_title: str = ""
    # A disc entry queued with 加入列队并做字幕: once it ends, every MKV it
    # wrote is queued for translation with this request (video_path filled
    # in per file), under this entry's settings snapshot — frozen when the
    # button was pressed, like everything else in the queue.
    then: Optional[JobRequest] = None
    # A disc entry queued with 加入列队并压制、做字幕: every MKV it wrote is
    # queued for re-encoding first, right behind it, with this request
    # (source filled in per file) — and `then` travels on to those entries,
    # so each encode, once done, queues its own subtitles.
    then_encode: Optional[EncodeRequest] = None
    # the entry this one was queued by (see `then` / `then_encode`), and
    # what that was: "disc" or "encode". Empty kind on an entry from before
    # encodes existed means a disc.
    origin: str = ""
    origin_kind: str = ""


class QueueEntryView(BaseModel):
    """A queue entry as the UI sees it: no snapshot, plus the live stage.

    The snapshot runs to a couple of kilobytes; the list is polled every
    few seconds, so the full text lives behind /api/queue/{id}/settings and
    only its fingerprint travels here.
    """

    id: str
    kind: str = "job"
    status: str = "queued"
    title: str = ""
    summary: str = ""          # "ja → 简体中文 · 双语"
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    job_id: str = ""
    error: str = ""
    note: str = ""
    settings_hash: str = ""
    settings_differs: bool = False
    interrupted: int = 0
    # live, and only while the job is still in memory
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    job_live: bool = False
    has_log: bool = False
    result_srt: str = ""
    result_srt_original: str = ""
    result_video: str = ""
    result_in_place: bool = False
    result_files: list[str] = []
    group_id: str = ""
    group_title: str = ""


class CpuYieldStatus(BaseModel):
    """压制让路此刻的状态（services/cpuyield.py），列队页显示用。"""
    supported: bool = True           # 这个系统读得到整机的 CPU 占用
    active: bool = False             # 有压制正在受它管
    limited: bool = False            # 正在限速
    others: Optional[float] = None   # 其他程序的 CPU 占用（0–1），还没量到时为空
    cap: float = 1.0                 # 压制此刻最多能用的份额（1 = 不限）


class MemoryStatus(BaseModel):
    """压制的内存保护（services/memguard.py），列队页显示用。单位 GB。"""
    supported: bool = True           # 这个系统读得到内存用量
    total_gb: float = 0.0            # 物理内存
    available_gb: float = 0.0        # 整机此刻可用
    floor_gb: float = 0.0            # 整机可用低于它就停下压制
    limit_gb: float = 0.0            # 压制自己最多用多少（生效的值）
    auto_gb: float = 0.0             # 「自动」对应的值：物理内存的一半
    active: bool = False             # 有压制正在受它管
    used_gb: float = 0.0             # 这次压制此刻用了多少
    peak_gb: float = 0.0


class QueueView(BaseModel):
    paused: bool = False
    cpu_yield: bool = False          # 列队页的「压制让出 CPU」
    cpu_status: CpuYieldStatus = CpuYieldStatus()
    memory_limit_gb: float = 0.0     # 列队页的「压制内存上限」，0 = 自动
    memory_status: MemoryStatus = MemoryStatus()
    worker_alive: bool = False
    active_id: str = ""        # answers "why is nothing running"
    settings_hash: str = ""    # fingerprint of the CURRENT settings
    entries: list[QueueEntryView] = []


DiscEnqueueRequest.model_rebuild()
DiscBatchRequest.model_rebuild()

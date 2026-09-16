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


class AppSettings(BaseModel):
    # temp working dir for intermediate files; empty = platform cache dir.
    # only its "jobs" subdirectory is managed (and wiped on startup)
    work_dir: str = ""
    # where downloaded whisper models are stored; empty = HuggingFace default
    # cache (~/.cache/huggingface/hub). changing it does NOT move old models
    model_cache_dir: str = ""
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
    output_mode: Literal["bilingual", "translation_only", "original_only"] = "bilingual"
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
    # see JobRequest.output_mode — original_only 跳过翻译，只输出原文
    output_mode: Literal["bilingual", "translation_only", "original_only"] = "bilingual"
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


JobStage = Literal[
    "pending",
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
    # embed mode: the new video carrying the subtitle track. Empty otherwise,
    # so the UI can tell the two outcomes apart from this field alone.
    video_filename: str = ""


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
                               # newer version should report itself, not fail
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    title: str = ""            # survives even when the rest cannot be parsed
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    job_id: str = ""
    request: Optional[JobRequest] = None
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
    result_video: str = ""
    result_in_place: bool = False
    # Files enqueued from one directory share these, so the UI can collapse
    # them into a single row instead of drowning the list.
    group_id: str = ""
    group_title: str = ""


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
    result_video: str = ""
    result_in_place: bool = False
    group_id: str = ""
    group_title: str = ""


class QueueView(BaseModel):
    paused: bool = False
    worker_alive: bool = False
    active_id: str = ""        # answers "why is nothing running"
    settings_hash: str = ""    # fingerprint of the CURRENT settings
    entries: list[QueueEntryView] = []

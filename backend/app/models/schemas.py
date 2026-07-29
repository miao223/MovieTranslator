"""Pydantic models shared by settings, jobs and the API layer."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- settings


class LLMSettings(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    # vision model for on-screen text translation; empty = use `model`
    # (needed because strong text models like DeepSeek have no vision)
    vision_model: str = ""
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
    # source-language hint fed to whisper as `initial_prompt`: proper nouns
    # it keeps mis-hearing (character names above all). MUST be written in
    # the spoken language — a prompt in another language drags the whole
    # transcript toward that language, which is why the plot synopsis and
    # the translation glossary are deliberately NOT reused here.
    initial_prompt: str = ""


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


class JobRequest(BaseModel):
    video_path: str
    # container stream index of the audio track to transcribe; None = the
    # track flagged default, else the first one
    audio_track: Optional[int] = None
    # fallback when audio_track is None (batch jobs, where indices differ per
    # file): prefer a track tagged with this language, e.g. "jpn"
    audio_language: str = ""
    source_language: str = "auto"  # whisper language code or "auto"
    target_language: str = "简体中文"
    synopsis: str = ""  # optional plot synopsis to steer the translation
    output_mode: Literal["bilingual", "translation_only"] = "bilingual"
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
    source_language: str = "auto"
    target_language: str = "简体中文"
    synopsis: str = ""  # shared synopsis is useful for TV series batches
    output_mode: Literal["bilingual", "translation_only"] = "bilingual"


JobStage = Literal[
    "pending",
    "extracting",
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

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
    # lines translated per output batch (limited by the model's max output tokens)
    batch_size: int = Field(80, ge=1, le=500)
    # approximate context window of the model, in tokens; beyond this the
    # translator falls back to sliding-window chunking
    context_limit: int = Field(100_000, ge=1_000)


class NetworkSettings(BaseModel):
    proxy_url: str = ""  # e.g. http://127.0.0.1:7890
    llm_via_proxy: bool = False
    model_download_via_proxy: bool = False


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
    vad_min_speech_ms: int = Field(250, ge=0, le=5000)
    vad_min_silence_ms: int = Field(2000, ge=100, le=10000)
    vad_speech_pad_ms: int = Field(400, ge=0, le=3000)


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
    # switches targeting known ASR weaknesses
    fix_asr_errors: bool = True      # homophone / mis-recognition correction
    link_fragments: bool = True      # cross-line coherence for fragmented lines
    normalize_loanwords: bool = True # katakana / transliterated loanword handling
    limit_length: bool = True        # keep translation subtitle-length friendly
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
    llm: LLMSettings = LLMSettings()
    asr: ASRSettings = ASRSettings()
    subtitle: SubtitleSettings = SubtitleSettings()
    prompts: PromptSettings = PromptSettings()
    network: NetworkSettings = NetworkSettings()


# ---------------------------------------------------------------- jobs


class FrameTask(BaseModel):
    """One on-screen-text translation point (画面翻译)."""

    time: str  # "1:23:45" / "23:45" / "85"
    note: str = ""  # hint for the vision model, e.g. "手机短信内容"
    duration: float = Field(5.0, ge=1.0, le=60.0)  # cue display seconds


class JobRequest(BaseModel):
    video_path: str
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
    source_language: str = "auto"
    target_language: str = "简体中文"
    synopsis: str = ""  # shared synopsis is useful for TV series batches
    output_mode: Literal["bilingual", "translation_only"] = "bilingual"


JobStage = Literal[
    "pending",
    "extracting",
    "transcribing",
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

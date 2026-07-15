"""On-screen text translation via a vision-capable OpenAI-compatible model."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from app.models.schemas import LLMSettings, NetworkSettings
from app.services.translator import make_openai_client

NO_TEXT_MARKER = "[无文字]"


def translate_frame(
    image_path: str | Path,
    target_language: str,
    note: str = "",
    llm: Optional[LLMSettings] = None,
    network: Optional[NetworkSettings] = None,
    client=None,  # injectable for tests
) -> Optional[str]:
    """Extract and translate visible text in the frame.

    Returns the translated text, or None when the model reports no
    meaningful text. Raises on API errors — the pipeline catches per task.
    """
    if client is None:
        client = make_openai_client(llm, network)
    model = (llm.vision_model.strip() or llm.model) if llm else "gpt-4o-mini"

    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    prompt = (
        "你是字幕组的画面文字翻译。这是电影某一时刻的画面截图，"
        "画面中可能有手机屏幕、纸质文件、电视画面、招牌等文字内容。"
        f"请提取其中对剧情有意义的文字并翻译成{target_language}，"
        "只输出译文本身（可多行），简洁明了，不要任何解释。"
        f"如果画面中没有值得翻译的文字，只输出 {NO_TEXT_MARKER}。"
    )
    if note.strip():
        prompt += f"\n用户备注（画面内容提示）：{note.strip()}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.2,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content or NO_TEXT_MARKER in content:
        return None
    return content

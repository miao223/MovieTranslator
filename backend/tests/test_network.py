import os

from app.models.schemas import LLMSettings, NetworkSettings
from app.services.asr import is_local_model_dir, proxy_env
from app.services.translator import make_openai_client


def test_proxy_env_sets_and_restores(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://old:1")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    net = NetworkSettings(proxy_url="http://p:7890", model_download_via_proxy=True)
    with proxy_env(net):
        assert os.environ["HTTPS_PROXY"] == "http://p:7890"
        assert os.environ["HTTP_PROXY"] == "http://p:7890"
    assert os.environ["HTTPS_PROXY"] == "http://old:1"
    assert "HTTP_PROXY" not in os.environ


def test_proxy_env_noop_when_disabled():
    monkey_before = os.environ.get("HTTPS_PROXY")
    net = NetworkSettings(proxy_url="http://p:7890", model_download_via_proxy=False)
    with proxy_env(net):
        assert os.environ.get("HTTPS_PROXY") == monkey_before
    with proxy_env(None):
        assert os.environ.get("HTTPS_PROXY") == monkey_before


def test_make_openai_client_without_proxy():
    client = make_openai_client(LLMSettings(base_url="http://x/v1", api_key="k"))
    assert str(client.base_url).startswith("http://x/v1")


def test_make_openai_client_with_proxy_builds_proxied_http_client():
    net = NetworkSettings(proxy_url="http://127.0.0.1:7890", llm_via_proxy=True)
    client = make_openai_client(LLMSettings(base_url="http://x/v1", api_key="k"), net)
    # the underlying httpx client must carry a proxy-mounted transport
    mounts = getattr(client._client, "_mounts", {})
    assert mounts, "expected proxy mounts on the httpx client"


def test_wrap_cuda_error_maps_dll_failure():
    from app.models.schemas import ASRSettings
    from app.services.asr import _wrap_cuda_error

    raw = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    wrapped = _wrap_cuda_error(raw, ASRSettings(device="cuda"))
    assert wrapped is not raw
    assert "CUDA 运行库加载失败" in str(wrapped)
    assert "cublas64_12.dll" in str(wrapped)
    # unrelated errors pass through untouched
    other = ValueError("boom")
    assert _wrap_cuda_error(other, ASRSettings(device="cuda")) is other
    # cpu device never wraps
    assert _wrap_cuda_error(raw, ASRSettings(device="cpu")) is raw


def test_model_alias_resolution():
    from app.services.asr import resolve_model
    from app.services.model_download import _repo_id

    assert resolve_model("large-v2") == "large-v2"
    assert resolve_model("kotoba-whisper-v2.0") == "kotoba-tech/kotoba-whisper-v2.0-faster"
    assert _repo_id("kotoba-whisper-v2.0") == "kotoba-tech/kotoba-whisper-v2.0-faster"
    assert _repo_id("CrisperWhisper") == "nyrahealth/faster_CrisperWhisper"
    assert _repo_id("large-v2") == "Systran/faster-whisper-large-v2"
    import pytest as _pytest

    with _pytest.raises(ValueError):
        _repo_id("not-a-model")


def test_is_local_model_dir(tmp_path):
    assert not is_local_model_dir(str(tmp_path))          # empty dir
    (tmp_path / "model.bin").write_bytes(b"x")
    assert is_local_model_dir(str(tmp_path))
    assert not is_local_model_dir(str(tmp_path / "nope"))  # missing dir


# ------------------------------------- what the "test connection" button says


class _Msg:
    def __init__(self, content, reasoning=""):
        self.content = content
        self.reasoning_content = reasoning


def _response(content, reasoning="", reasoning_tokens=0):
    class Details:
        pass

    class Usage:
        pass

    class Resp:
        pass

    details = Details()
    details.reasoning_tokens = reasoning_tokens
    usage = Usage()
    usage.completion_tokens_details = details
    resp = Resp()
    choice = Details()
    choice.message = _Msg(content, reasoning)
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _describe(disable_thinking, resp, accepted=True):
    from app.api.routes import _describe_thinking
    from app.models.schemas import LLMSettings

    llm = LLMSettings(model="m", disable_thinking=disable_thinking)
    return _describe_thinking(llm, resp, resp.choices[0].message, accepted)


def test_thinking_reported_as_successfully_off():
    out = _describe(True, _response("5"))
    assert out["level"] == "success"
    assert "已成功关闭" in out["text"]


def test_thinking_reported_when_the_model_thought_anyway():
    out = _describe(True, _response("5", reasoning="let me see...", reasoning_tokens=42))
    assert out["level"] == "warning"
    assert "仍返回了思考内容" in out["text"]
    assert out["reasoning_tokens"] == 42


def test_thinking_reported_when_the_provider_refused_the_parameter():
    out = _describe(True, _response("5"), accepted=False)
    assert out["level"] == "warning"
    assert "不认识关闭思考的参数" in out["text"]


def test_thinking_reported_as_on_when_the_switch_is_off():
    out = _describe(False, _response("5", reasoning="hmm", reasoning_tokens=7))
    assert out["level"] == "info"
    assert "开启" in out["text"]

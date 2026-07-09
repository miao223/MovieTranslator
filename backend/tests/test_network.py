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


def test_is_local_model_dir(tmp_path):
    assert not is_local_model_dir(str(tmp_path))          # empty dir
    (tmp_path / "model.bin").write_bytes(b"x")
    assert is_local_model_dir(str(tmp_path))
    assert not is_local_model_dir(str(tmp_path / "nope"))  # missing dir

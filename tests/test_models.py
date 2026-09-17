from types import SimpleNamespace

import torch

from agentic_rag.models import ModelRuntime


def test_openai_compatible_backend_uses_configured_endpoint_and_usage(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "回答 [1]"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.models.httpx.post", fake_post)
    settings = SimpleNamespace(
        llm_backend="openai_compatible",
        llm_api_base="http://vllm:8000/v1",
        llm_api_key="secret",
        llm_api_timeout_seconds=30,
        llm_model="Qwen/Qwen2.5-7B-Instruct",
    )
    runtime = ModelRuntime(settings)

    answer = runtime.generate("系统", "问题", 128)

    assert answer == "回答 [1]"
    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["max_tokens"] == 128
    assert runtime.prompt_tokens == 12
    assert runtime.completion_tokens == 4


def test_local_cuda_falls_back_to_float16_when_bfloat16_is_unsupported(monkeypatch):
    captured = {}

    def fake_from_pretrained(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agentic_rag.models.torch.cuda.is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        "agentic_rag.models.AutoModelForCausalLM.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(ModelRuntime, "_resolve", lambda self, model_id: "/models/qwen")
    settings = SimpleNamespace(device="cuda", llm_model="Qwen/Qwen2.5-1.5B-Instruct")

    runtime = ModelRuntime(settings)
    _ = runtime.llm

    assert captured["torch_dtype"] is torch.float16
    assert captured["device_map"] == "auto"

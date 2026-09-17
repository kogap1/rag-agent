from __future__ import annotations

from functools import cached_property
from threading import local
from typing import Iterator

import httpx
import torch
from modelscope import snapshot_download
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import Settings


class ModelRuntime:
    """Lazy local-model runtime so indexing does not load the LLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._usage = local()
        self.reset_usage()

    def reset_usage(self) -> None:
        self._usage.prompt_tokens = 0
        self._usage.completion_tokens = 0

    @property
    def prompt_tokens(self) -> int:
        return int(getattr(self._usage, "prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(getattr(self._usage, "completion_tokens", 0))

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def device(self) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve(self, model_id: str) -> str:
        organization, _, model_name = model_id.partition("/")
        candidates = [
            self.settings.model_cache_dir / organization / model_name,
            self.settings.model_cache_dir / organization / model_name.replace(".", "___"),
        ]
        for local in candidates:
            if local.exists():
                return str(local)
        return snapshot_download(model_id, cache_dir=str(self.settings.model_cache_dir))

    @cached_property
    def embedding(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_path = self._resolve(self.settings.embedding_model)
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @cached_property
    def reranker(self) -> CrossEncoder | None:
        if not self.settings.enable_reranker:
            return None
        model_path = self._resolve(self.settings.reranker_model)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        return CrossEncoder(model_path, device=self.device, model_kwargs={"torch_dtype": dtype})

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self._resolve(self.settings.llm_model), trust_remote_code=True)

    @cached_property
    def llm(self):
        if self.device == "cuda":
            supports_bf16 = bool(
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        else:
            dtype = torch.float32
        return AutoModelForCausalLM.from_pretrained(
            self._resolve(self.settings.llm_model),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    def generate(self, system: str, user: str, max_new_tokens: int = 1024) -> str:
        if self.settings.llm_backend == "openai_compatible":
            return self._generate_openai_compatible(system, user, max_new_tokens)
        if self.settings.llm_backend != "local":
            raise ValueError(f"不支持的 LLM_BACKEND：{self.settings.llm_backend}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        output = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs.input_ids.shape[1] :]
        self._usage.prompt_tokens = self.prompt_tokens + int(inputs.input_ids.shape[1])
        self._usage.completion_tokens = self.completion_tokens + int(generated.shape[0])
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_openai_compatible(self, system: str, user: str, max_new_tokens: int) -> str:
        url = f"{self.settings.llm_api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        response = httpx.post(
            url,
            headers=headers,
            json={
                "model": self.settings.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_new_tokens,
                "stream": False,
            },
            timeout=self.settings.llm_api_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            answer = str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible 服务返回格式不合法") from exc
        usage = payload.get("usage") or {}
        self._usage.prompt_tokens = self.prompt_tokens + int(usage.get("prompt_tokens") or 0)
        self._usage.completion_tokens = self.completion_tokens + int(usage.get("completion_tokens") or 0)
        return answer

    def stream(self, system: str, user: str, max_new_tokens: int = 1024) -> Iterator[str]:
        if self.settings.llm_backend == "openai_compatible":
            yield self._generate_openai_compatible(system, user, max_new_tokens)
            return
        from threading import Thread

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
        Thread(target=self.llm.generate, kwargs=kwargs, daemon=True).start()
        yield from streamer

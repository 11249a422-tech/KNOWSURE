"""The pretrained Small Language Model: via the Gemini API, or on-device via Ollama or Hugging Face transformers."""
from __future__ import annotations

import threading
import time
from typing import Protocol

import httpx

from .config import Settings

CLOSED_BOOK_INSTRUCTION = "Answer the question in one short sentence."
GROUNDED_INSTRUCTION = ("Answer the question using only the numbered context passages. "
                        "If the context does not contain the answer, reply exactly: I don't know.")
RETRY_STATUSES = (429, 500, 502, 503, 504)  # rate limit and temporary server errors


class SLMError(Exception):
    """The SLM call failed (network, quota, invalid model, blocked output...)."""


class SLMNotConfigured(SLMError):
    """No API key is set."""


class Completer(Protocol):
    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str:
        """Send one prompt, return the model's final text."""


class Generator(Completer, Protocol):
    def generate(self, question: str, context: list[str] | None = None) -> str:
        """Answer the question; with context, answer only from it (used by the SLM + RAG baseline)."""


def build_prompt(question: str, context: list[str] | None = None) -> str:
    # Some Gemma models on the Gemini API reject a separate system instruction, so it goes in the prompt.
    if context:
        passages = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(context))
        return f"{GROUNDED_INSTRUCTION}\n\nContext:\n{passages}\n\nQuestion: {question}"
    return f"{CLOSED_BOOK_INSTRUCTION}\n\nQuestion: {question}"


class _PromptGenerator:
    """generate() in terms of complete(), shared by every provider."""

    def generate(self, question: str, context: list[str] | None = None) -> str:
        return self.complete(build_prompt(question, context))

    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str:  # pragma: no cover
        raise NotImplementedError


def _post_with_retries(client: httpx.Client, url: str, retries: int, delay: float, name: str, **kwargs) -> httpx.Response:
    for attempt in range(1 + retries):
        try:
            response = client.post(url, **kwargs)
        except httpx.HTTPError as exc:
            if attempt < retries:
                time.sleep(delay)
                continue
            raise SLMError(f"Could not reach {name} ({exc.__class__.__name__}).") from exc
        # Temporary failures ("Internal error encountered", per-minute rate limits): wait and try again.
        if response.status_code in RETRY_STATUSES and attempt < retries:
            time.sleep(delay * (2 if response.status_code == 429 else 1) * (attempt + 1))
            continue
        return response
    raise AssertionError("unreachable")


class GeminiGenerator(_PromptGenerator):
    provider = "gemini"

    def __init__(self, api_key: str, model: str, base_url: str, max_output_tokens: int, timeout: float,
                 thinking_level: str = "", client: httpx.Client | None = None, retries: int = 2,
                 retry_delay: float = 1.0):
        self.retries = retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str:
        """Send one prompt and return the model's final text (reasoning parts removed)."""
        if not self.api_key:
            raise SLMNotConfigured("No Gemini API key. Put GEMINI_API_KEY=... in the backend's .env file.")
        config = {"temperature": 0, "maxOutputTokens": max_output_tokens or self.max_output_tokens}
        if self.thinking_level:
            # Reasoning models think before answering; "minimal" keeps answers fast and within the token limit.
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        # The key goes in a header, never in the URL, so it can't leak into logs or error messages.
        response = _post_with_retries(self._client, self.url, self.retries, self.retry_delay, "the Gemini API",
                                      headers={"x-goog-api-key": self.api_key}, json=body)
        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.reason_phrase
            raise SLMError(f"Gemini API error {response.status_code} ({self.model}): {message}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise SLMError(f"Gemini returned no answer ({reason}).")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        # Skip the model's reasoning ("thought") parts: only the final answer is the candidate.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            raise SLMError(f"Gemini returned an empty answer ({candidates[0].get('finishReason', 'unknown')}).")
        return text


class OllamaGenerator(_PromptGenerator):
    """On-device SLM served by Ollama (https://ollama.com) through its OpenAI-compatible API."""
    provider = "ollama"

    def __init__(self, base_url: str, model: str, max_output_tokens: int, timeout: float = 120.0,
                 client: httpx.Client | None = None, retries: int = 1, retry_delay: float = 1.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_output_tokens or self.max_output_tokens}
        response = _post_with_retries(self._client, self.url, self.retries, self.retry_delay,
                                      f"Ollama at {self.url} (is `ollama serve` running?)", json=body)
        if response.status_code != 200:
            raise SLMError(f"Ollama error {response.status_code}: {response.text[:200]} "
                           f"(did you run `ollama pull {self.model}`?)")
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SLMError("Ollama returned an unexpected response.") from exc
        if not text:
            raise SLMError("Ollama returned an empty answer.")
        return text


class HFGenerator(_PromptGenerator):
    """On-device SLM run in-process with Hugging Face transformers. Loads on first use."""
    provider = "hf"

    def __init__(self, model_name: str, max_output_tokens: int):
        self.model = model_name
        self.max_output_tokens = max_output_tokens
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()  # one generation at a time on local hardware

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype="auto")
        self._model.eval()

    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            text = self._tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                       add_generation_prompt=True)
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=max_output_tokens or self.max_output_tokens,
                                              do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
        answer = self._tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not answer:
            raise SLMError("The local model returned an empty answer.")
        return answer


def build_generator(settings: Settings) -> GeminiGenerator | OllamaGenerator | HFGenerator:
    if settings.slm_provider == "ollama":
        return OllamaGenerator(settings.ollama_base_url, settings.ollama_model, settings.max_output_tokens)
    if settings.slm_provider == "hf":
        return HFGenerator(settings.hf_model, settings.max_output_tokens)
    return build_gemini(settings, settings.slm_model)


def build_gemini(settings: Settings, model: str, timeout: float | None = None, retries: int = 2) -> GeminiGenerator:
    return GeminiGenerator(settings.gemini_api_key, model, settings.gemini_base_url, settings.max_output_tokens,
                           timeout or settings.request_timeout, settings.thinking_level, retries=retries)

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot satisfy a request."""


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))


class OllamaClient:
    """Small HTTP client for local Ollama chat and embedding endpoints."""

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def _request_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        base = self.config.base_url.rstrip("/")
        url = f"{base}{path}"
        body = None
        headers: dict[str, str] = {}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} error for {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.base_url}. "
                "Start it with `ollama serve` and verify the URL."
            ) from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            snippet = raw[:300].replace("\n", " ")
            raise OllamaError(f"Non-JSON response from Ollama: {snippet}") from exc

    def healthcheck(self) -> bool:
        try:
            data = self._request_json("GET", "/api/tags")
        except OllamaError:
            return False
        return isinstance(data.get("models", []), list)

    def list_models(self) -> list[str]:
        data = self._request_json("GET", "/api/tags")
        models = data.get("models", [])
        return [item.get("name", "") for item in models if item.get("name")]

    def assert_model_available(self, model_name: str) -> str:
        requested = (model_name or "").strip()
        if not requested:
            raise OllamaError("Model name is empty. Set OLLAMA_CHAT_MODEL and OLLAMA_EMBED_MODEL explicitly.")

        available_models = self.list_models()
        if requested in available_models:
            return requested
        joined = ", ".join(sorted(available_models)) or "<none>"
        raise OllamaError(
            f"Model `{requested}` not found in local Ollama models. "
            f"Available models: {joined}. Pull the exact model with `ollama pull {requested}`."
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.config.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        data = self._request_json("POST", "/api/chat", payload)
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Unexpected /api/chat response payload: {data}")
        return content.strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages=messages, model=model, temperature=temperature, json_mode=json_mode)

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        payload = {
            "model": model or self.config.embedding_model,
            "prompt": text,
        }
        data = self._request_json("POST", "/api/embeddings", payload)
        if isinstance(data.get("embedding"), list):
            return [float(v) for v in data["embedding"]]
        if isinstance(data.get("embeddings"), list) and data["embeddings"]:
            return [float(v) for v in data["embeddings"][0]]
        raise OllamaError(f"Unexpected /api/embeddings response payload: {data}")

    def batch_embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        return [self.embed(text=t, model=model) for t in texts]
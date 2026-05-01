from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

class OllamaError(RuntimeError):
	"""
	Raised when the local Ollama server cannot satisfy a request.

	Common causes:
	- Ollama service not running (start with `ollama serve`)
	- Incorrect base URL configuration
	- Requested model not available locally
	- Network connectivity issues
	- HTTP errors from Ollama API
	"""

@dataclass
class OllamaConfig:
    """
    Configuration for OllamaClient.
    Attributes:
        base_url (str): Base URL for Ollama API (e.g., "http://localhost:11434").
        chat_model (str): Default model for chat generation (e.g., "llama3:latest").
        embedding_model (str): Default model for embeddings (e.g., "nomic-embed-text:latest").
        timeout_seconds (float): Timeout for HTTP requests in seconds. Defaults to 600.
    """    
    
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))

class OllamaClient:
    """
    Minimal HTTP client for local Ollama chat and embedding endpoints.

    Provides a simple interface to interact with Ollama's /api/chat and /api/embeddings
    endpoints. Handles JSON parsing, error management, and timeouts.

    All methods raise OllamaError on communication or parsing failures.

    Attributes:
        config (OllamaConfig): Configuration with base URL, models, and timeout.

    Example:
        llm = OllamaClient(OllamaConfig(base_url="http://localhost:11434"))
        if llm.healthcheck():
            response = llm.generate(prompt="Hello", system="You are helpful.")
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def _request_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Make an HTTP request to Ollama and parse JSON response.

        Args:
            method (str): HTTP method ("GET", "POST", etc.).
            path (str): API path relative to base URL (e.g., "/api/tags").
            payload (Optional[dict[str, Any]]): JSON payload for POST requests.

        Returns:
            dict[str, Any]: Parsed JSON response, or empty dict if response is empty.

        Raises:
            OllamaError: On HTTP errors, network issues, or JSON parse failures.
        """
        
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
        """
        Check if Ollama server is reachable and responding.

        Returns:
            bool: True if server is healthy, False otherwise.
        """
        
        try:
            data = self._request_json("GET", "/api/tags")
        except OllamaError:
            return False
        return isinstance(data.get("models", []), list)

    def list_models(self) -> list[str]:
        """
        List all available models in the Ollama instance.

        Returns:
            list[str]: Names of available models.

        Raises:
            OllamaError: If the request fails.
        """
        
        data = self._request_json("GET", "/api/tags")
        models = data.get("models", [])
        return [item.get("name", "") for item in models if item.get("name")]

    def assert_model_available(self, model_name: str) -> str:
        """
        Assert that a model is available locally, raise error if not.

        Args:
            model_name (str): Name of the model to check.

        Returns:
            str: The model name if available.

        Raises:
            OllamaError: If model is not available or model_name is empty.
        """
        
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
        """
        Send a multi-turn chat request to Ollama.

        Args:
            messages (list[dict[str, str]]): List of messages with "role" and "content" keys.
            model (Optional[str]): Model to use (defaults to config.chat_model).
            temperature (float): Sampling temperature [0.0, 2.0]. Defaults to 0.2.
            json_mode (bool): Request JSON-formatted output. Defaults to False.

        Returns:
            str: The assistant's response text.

        Raises:
            OllamaError: If request fails or response is malformed.
        """
        
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
        """
        Generate a response from a single prompt (convenience wrapper around chat).

        Args:
            prompt (str): The user prompt.
            system (Optional[str]): System prompt to set context.
            model (Optional[str]): Model to use (defaults to config.chat_model).
            temperature (float): Sampling temperature. Defaults to 0.2.
            json_mode (bool): Request JSON-formatted output. Defaults to False.

        Returns:
            str: The generated response.

        Raises:
            OllamaError: If request fails.
        """
        
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages=messages, model=model, temperature=temperature, json_mode=json_mode)

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        Generate an embedding vector for text.

        Args:
            text (str): The text to embed.
            model (Optional[str]): Embedding model to use (defaults to config.embedding_model).

        Returns:
            list[float]: The embedding vector.

        Raises:
            OllamaError: If embedding generation fails.
        """
        
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
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts (list[str]): List of texts to embed.
            model (Optional[str]): Embedding model to use.

        Returns:
            list[list[float]]: List of embedding vectors.

        Raises:
            OllamaError: If any embedding fails.
        """
        
        return [self.embed(text=t, model=model) for t in texts]
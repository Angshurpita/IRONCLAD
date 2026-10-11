"""LLM inference providers for Google Gemma 4 26B A4B with OpenRouter and Ollama."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from ironclad.config import IroncladSettings
from ironclad.errors import ConfigurationError, InferenceError

logger = logging.getLogger("ironclad.inference")


class BaseInferenceProvider(ABC):
    """Abstract interface for LLM inference providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the inference service."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Send prompt to LLM and return raw text response."""
        pass


class OpenRouterGemmaProvider(BaseInferenceProvider):
    """
    Hosted inference provider connecting to OpenRouter for Gemma 4 26B A4B.
    Implements bounded exponential backoff retries, privacy warnings, and timeout protection.
    """

    def __init__(self, settings: IroncladSettings):
        self.settings = settings
        if not settings.openrouter_api_key or not settings.openrouter_api_key.get_secret_value().strip():
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not configured.",
                hint="Provide OPENROUTER_API_KEY in your environment or in the .env file.",
            )
        self.api_key = settings.openrouter_api_key.get_secret_value()
        self.base_url = settings.openrouter_base_url.rstrip("/")
        model = settings.openrouter_model
        if model in ("google/gemma-4-26b-a4b", "google/gemma-4-26b"):
            model = "google/gemma-4-26b-a4b-it"
        elif model in ("google/gemma-4-26b-a4b:free", "google/gemma-4-26b:free"):
            model = "google/gemma-4-26b-a4b-it:free"
        elif model == "google/gemma-4-31b":
            model = "google/gemma-4-31b-it"
        elif model == "google/gemma-4-31b:free":
            model = "google/gemma-4-31b-it:free"
        self.model = model
        self.timeout = settings.http_timeout_seconds
        self.max_retries = max(settings.max_retries, 4)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/ironclad-rag/ironclad",
            "X-Title": "Ironclad RAG CLI",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }

        delay = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)

                    # Immediate non-retriable auth errors
                    if response.status_code in (401, 403):
                        raise InferenceError(
                            f"OpenRouter authentication failed (HTTP {response.status_code}): {response.text}",
                            hint="Verify your OPENROUTER_API_KEY is valid and has sufficient credits.",
                        )

                    # Retriable rate limits or transient 5xx errors
                    if response.status_code in (429, 500, 502, 503, 504):
                        last_error = Exception(f"HTTP {response.status_code}: {response.text}")
                        logger.warning(
                            f"OpenRouter HTTP {response.status_code} (attempt {attempt}/{self.max_retries}). Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= 2.0
                        continue

                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise InferenceError("OpenRouter response contained no choices.")
                    content = choices[0].get("message", {}).get("content", "")
                    return content

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                logger.warning(f"Network error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise InferenceError(
                        f"Inference request failed after {self.max_retries} attempts: {e}",
                        hint="Check internet connectivity and proxy settings.",
                    ) from e
            except InferenceError:
                raise
            except Exception as e:
                raise InferenceError(f"Unexpected error communicating with OpenRouter: {e}") from e

        # If free endpoint failed due to upstream public rate limits, attempt standard model fallback
        if ":free" in self.model:
            fallback_model = self.model.replace(":free", "")
            logger.info(
                f"OpenRouter free pool ({self.model}) was rate-limited upstream. Routing to {fallback_model}..."
            )
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    fb_payload = dict(payload)
                    fb_payload["model"] = fallback_model
                    response = client.post(url, headers=headers, json=fb_payload)
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
            except Exception:
                pass

        raise InferenceError(
            f"OpenRouter failed after {self.max_retries} attempts: {last_error}",
            hint="The free tier pool may be temporarily rate-limited upstream. Please retry shortly or check OpenRouter status.",
        )


class OllamaGemmaProvider(BaseInferenceProvider):
    """
    Local inference provider connecting to local Ollama runtime.
    Validates connectivity and model availability without automatic downloading.
    """

    def __init__(self, settings: IroncladSettings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = settings.http_timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def _verify_availability(self) -> None:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    raise InferenceError(
                        f"Ollama server returned HTTP {resp.status_code}.",
                        hint=f"Ensure Ollama is running at {self.base_url}",
                    )
                models_data = resp.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                if not any(self.model in m for m in model_names):
                    raise InferenceError(
                        f"Model '{self.model}' is not installed in local Ollama.",
                        hint=(
                            f"Run: ollama pull {self.model}\n"
                            "Note: Gemma 4 26B requires substantial VRAM/RAM (~16-32GB)."
                        ),
                    )
        except httpx.RequestError as e:
            raise InferenceError(
                f"Cannot connect to Ollama at {self.base_url}: {e}",
                hint="Start Ollama locally with `ollama serve`.",
            ) from e

    def generate(self, prompt: str, system_prompt: str) -> str:
        self._verify_availability()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            raise InferenceError(f"Error querying local Ollama runtime: {e}") from e


class MockGemmaProvider(BaseInferenceProvider):
    """
    Mock inference provider for automated tests and offline evaluation.
    Returns deterministic structured responses.
    """

    def __init__(
        self,
        canned_answer: str = "Mock answer based on evidence.",
        citations: Optional[list] = None,
        insufficient_evidence: bool = False,
    ):
        self._canned_answer = canned_answer
        self._citations = citations or []
        self._insufficient_evidence = insufficient_evidence

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-gemma-4-26b"

    def generate(self, prompt: str, system_prompt: str) -> str:
        return json.dumps(
            {
                "answer": self._canned_answer,
                "citations": self._citations,
                "insufficient_evidence": self._insufficient_evidence,
            }
        )


def get_inference_provider(
    settings: IroncladSettings,
    mock_provider: Optional[BaseInferenceProvider] = None,
) -> BaseInferenceProvider:
    """Factory creating configured inference provider."""
    if mock_provider:
        return mock_provider

    provider_name = settings.inference_provider.lower().strip()
    if provider_name in ("openrouter", "openrouter.ai"):
        return OpenRouterGemmaProvider(settings)
    elif provider_name == "ollama":
        return OllamaGemmaProvider(settings)
    else:
        raise ConfigurationError(
            f"Unsupported inference provider '{provider_name}'.",
            hint="Supported providers: 'openrouter', 'ollama'.",
        )

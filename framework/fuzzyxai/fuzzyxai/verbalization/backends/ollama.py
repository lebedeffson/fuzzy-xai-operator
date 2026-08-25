from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fuzzyxai.verbalization.contracts import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnreachableError,
    InvalidBackendResponseError,
    ModelNotFoundError,
)

# Recommended default per the project's own guidance: small, multilingual,
# reasonable local footprint. Deliberately not "latest" — that tag is not a
# reproducible reference (its contents change over time).
# https://ollama.com/library/qwen3
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 30.0
MAX_RESPONSE_BYTES = 1_000_000  # 1 MB — a verbalized explanation is a short paragraph, not a novel


def _env_default(name: str, fallback: str) -> str:
    return os.environ.get(name) or fallback


class OllamaBackend:
    """Local model access over Ollama's HTTP API, stdlib-only (no new dependency).

    Requires the user to run ``ollama serve`` and pull a model themselves —
    this framework never bundles, downloads, or auto-installs a model or the
    Ollama runtime. See https://docs.ollama.com/linux, .../macos for the
    per-OS install steps (they are genuinely different — Linux uses an
    install script/CLI, macOS/Windows a desktop app/installer — do not tell
    users "the same command works everywhere").

    Model and host default from ``FUZZYXAI_OLLAMA_MODEL``/``FUZZYXAI_OLLAMA_HOST``
    when set, falling back to ``qwen3:1.7b`` / ``http://localhost:11434``;
    an explicit constructor argument always wins over both.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model or _env_default("FUZZYXAI_OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or _env_default("FUZZYXAI_OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.timeout = float(timeout)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            if exc.code == 404 and "model" in detail.lower():
                raise ModelNotFoundError(f"model '{self.model}' was not found on {self.host} — run `ollama pull {self.model}` first") from exc
            raise BackendHTTPError(exc.code, detail) from exc
        except TimeoutError as exc:
            raise BackendTimeoutError(f"Ollama at {self.host} did not respond within {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            raise BackendUnreachableError(f"could not reach Ollama at {self.host}: {exc.reason}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InvalidBackendResponseError(f"Ollama response exceeded the {MAX_RESPONSE_BYTES}-byte limit")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidBackendResponseError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise InvalidBackendResponseError("Ollama response was valid JSON but not an object")
        return parsed

    def generate(self, prompt: str, *, response_schema: Mapping[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # avoid thinking-mode preamble text on models that support it
            "options": {"temperature": 0},
        }
        if response_schema is not None:
            payload["format"] = dict(response_schema)
        body = self._post("/api/generate", payload)
        if "error" in body:
            message = str(body["error"])
            if "not found" in message.lower():
                raise ModelNotFoundError(f"model '{self.model}' was not found on {self.host}: {message}")
            raise InvalidBackendResponseError(f"Ollama returned an error: {message}")
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise InvalidBackendResponseError("Ollama returned an empty 'response' field")
        return text.strip()

    def check(self) -> OllamaHealth:
        """Report whether Ollama is reachable and the configured model is present.

        Never installs, downloads, or pulls anything — read-only diagnostic.
        """

        request = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError:
            return OllamaHealth(reachable=False, model_present=False, models=(), detail=f"timed out after {self.timeout}s")
        except urllib.error.URLError as exc:
            return OllamaHealth(reachable=False, model_present=False, models=(), detail=str(exc.reason))
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return OllamaHealth(reachable=True, model_present=False, models=(), detail=f"invalid /api/tags response: {exc}")
        models = tuple(str(item.get("name", "")) for item in parsed.get("models", []) if isinstance(item, Mapping))
        present = self.model in models or any(name.startswith(f"{self.model}:") for name in models)
        return OllamaHealth(reachable=True, model_present=present, models=models, detail=None)


@dataclass(frozen=True)
class OllamaHealth:
    reachable: bool
    model_present: bool
    models: tuple[str, ...]
    detail: str | None

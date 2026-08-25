from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


class VerbalizationBackendError(RuntimeError):
    """Base class for verbalization backend failures."""


class BackendUnreachableError(VerbalizationBackendError):
    """The backend server could not be reached at all (connection refused/DNS/etc.)."""


class BackendTimeoutError(VerbalizationBackendError):
    """The backend did not respond within the configured timeout."""


class ModelNotFoundError(VerbalizationBackendError):
    """The configured model is not available on the backend."""


class BackendHTTPError(VerbalizationBackendError):
    """The backend returned a non-success HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class InvalidBackendResponseError(VerbalizationBackendError):
    """The backend's response was not valid JSON, or didn't match the expected shape."""


@runtime_checkable
class VerbalizationBackend(Protocol):
    """One rephrasing engine. The framework depends only on this protocol.

    Implementations may call a local model (see ``backends.ollama``), an
    in-process model, or any other text generator. They must raise a
    ``VerbalizationBackendError`` subclass on failure rather than returning a
    partial or placeholder string — the caller (``SLMVerbalizer``) decides
    how to fall back, this layer must not hide or paper over the failure.

    ``response_schema``, when given, is a hint that the backend should
    constrain its output to match this JSON schema if it supports structured
    output (e.g. Ollama's ``format`` field). A backend that doesn't support
    this may ignore it — the caller still validates the parsed response
    itself and never trusts schema conformance alone.
    """

    def generate(self, prompt: str, *, response_schema: Mapping[str, Any] | None = None) -> str:
        ...

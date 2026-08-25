"""Ollama transport tests against a local mock HTTP server — no real Ollama needed.

Covers §7.4 of the explanation-output-layer spec: request shape (URL,
headers, body, stream/think/format flags), successful response parsing, and
every documented failure mode (model missing, server error, timeout,
malformed JSON) — all through typed exceptions, never a silent/blank result.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fuzzyxai.verbalization.backends.ollama import OllamaBackend
from fuzzyxai.verbalization.contracts import (
    BackendHTTPError,
    BackendTimeoutError,
    InvalidBackendResponseError,
    ModelNotFoundError,
)


class _RecordingServer:
    """A minimal HTTP server whose handler behavior is set per-test."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.next_response: tuple[int, bytes] = (200, json.dumps({"response": "ok"}).encode())
        self.delay_seconds = 0.0
        self._httpd = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # silence test output
                pass

            def do_POST(self) -> None:
                import time

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers.items()),
                        "body": json.loads(body.decode("utf-8")) if body else None,
                    }
                )
                if server.delay_seconds:
                    time.sleep(server.delay_seconds)
                status, payload = server.next_response
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                server.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": None})
                status, payload = server.next_response
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def server():
    instance = _RecordingServer()
    yield instance
    instance.close()


def test_request_shape_url_headers_and_flags(server) -> None:
    backend = OllamaBackend(model="qwen3:1.7b", host=server.host)
    backend.generate("explain this")

    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["path"] == "/api/generate"
    assert request["headers"]["Content-Type"] == "application/json"
    body = request["body"]
    assert body["model"] == "qwen3:1.7b"
    assert body["prompt"] == "explain this"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0


def test_response_schema_is_passed_as_format(server) -> None:
    backend = OllamaBackend(host=server.host)
    schema = {"type": "object", "properties": {"order": {"type": "array"}}}
    backend.generate("explain this", response_schema=schema)
    assert server.requests[0]["body"]["format"] == schema


def test_successful_response_is_parsed(server) -> None:
    server.next_response = (200, json.dumps({"response": "  the answer  "}).encode())
    backend = OllamaBackend(host=server.host)
    assert backend.generate("x") == "the answer"


def test_model_not_found_raises_typed_error(server) -> None:
    server.next_response = (404, json.dumps({"error": "model 'qwen3:1.7b' not found, try pulling it first"}).encode())
    backend = OllamaBackend(host=server.host)
    with pytest.raises(ModelNotFoundError):
        backend.generate("x")


def test_server_error_raises_typed_http_error(server) -> None:
    server.next_response = (500, b"internal error")
    backend = OllamaBackend(host=server.host)
    with pytest.raises(BackendHTTPError) as exc_info:
        backend.generate("x")
    assert exc_info.value.status == 500


def test_timeout_raises_typed_error(server) -> None:
    server.delay_seconds = 0.5
    backend = OllamaBackend(host=server.host, timeout=0.05)
    with pytest.raises(BackendTimeoutError):
        backend.generate("x")


def test_malformed_json_raises_typed_error(server) -> None:
    server.next_response = (200, b"{not valid json")
    backend = OllamaBackend(host=server.host)
    with pytest.raises(InvalidBackendResponseError):
        backend.generate("x")


def test_error_field_in_200_response_still_raises(server) -> None:
    # Ollama can return HTTP 200 with an "error" field for some failure modes.
    server.next_response = (200, json.dumps({"error": "something went wrong"}).encode())
    backend = OllamaBackend(host=server.host)
    with pytest.raises(InvalidBackendResponseError):
        backend.generate("x")


def test_empty_response_field_raises(server) -> None:
    server.next_response = (200, json.dumps({"response": "   "}).encode())
    backend = OllamaBackend(host=server.host)
    with pytest.raises(InvalidBackendResponseError):
        backend.generate("x")


def test_check_reports_reachable_and_model_present(server) -> None:
    server.next_response = (200, json.dumps({"models": [{"name": "qwen3:1.7b"}, {"name": "llama3.2:1b"}]}).encode())
    backend = OllamaBackend(model="qwen3:1.7b", host=server.host)
    health = backend.check()
    assert health.reachable is True
    assert health.model_present is True
    assert "qwen3:1.7b" in health.models


def test_check_reports_model_missing_without_installing_anything(server) -> None:
    server.next_response = (200, json.dumps({"models": [{"name": "llama3.2:1b"}]}).encode())
    backend = OllamaBackend(model="qwen3:1.7b", host=server.host)
    health = backend.check()
    assert health.reachable is True
    assert health.model_present is False


def test_check_reports_unreachable_server() -> None:
    backend = OllamaBackend(host="http://127.0.0.1:1", timeout=1.0)
    health = backend.check()
    assert health.reachable is False
    assert health.detail is not None

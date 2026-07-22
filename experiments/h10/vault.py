from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path


MAGIC = b"FXAI-H10-VAULT-v1\0"


def _stream(key: bytes, nonce: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(output[:length])


def create_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes()
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    path.chmod(0o600)
    return key


def seal(payload: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(16)
    ciphertext = bytes(left ^ right for left, right in zip(payload, _stream(key, nonce, len(payload)), strict=True))
    tag = hmac.new(key, MAGIC + nonce + ciphertext, hashlib.sha256).digest()
    return MAGIC + nonce + tag + ciphertext


def open_vault(payload: bytes, key: bytes) -> bytes:
    if not payload.startswith(MAGIC):
        raise ValueError("invalid H10 vault header")
    offset = len(MAGIC)
    nonce, tag, ciphertext = payload[offset : offset + 16], payload[offset + 16 : offset + 48], payload[offset + 48 :]
    expected = hmac.new(key, MAGIC + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("H10 vault integrity check failed")
    return bytes(left ^ right for left, right in zip(ciphertext, _stream(key, nonce, len(ciphertext)), strict=True))

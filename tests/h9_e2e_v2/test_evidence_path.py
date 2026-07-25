from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from fuzzyxai.evidence_path import (
    StaticArtifactCache,
    StaticManifest,
    audit_batch,
    merkle_root,
    tensor_digest,
)


def _manifest() -> StaticManifest:
    digest = hashlib.sha256(b"fixed").hexdigest()
    return StaticManifest("model", "1", digest, digest, digest, digest, digest, digest)


def test_binary_digest_is_layout_canonical_and_shape_sensitive() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert tensor_digest(values) == tensor_digest(np.ascontiguousarray(values))
    assert tensor_digest(values) != tensor_digest(values.reshape(2, 6))
    assert tensor_digest(values) != tensor_digest(values.astype(np.float64))


def test_digest_implementation_does_not_use_tolist() -> None:
    path = Path("framework/fuzzyxai/fuzzyxai/evidence_path/digest.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "tolist" not in attributes


def test_merkle_root_is_order_independent_by_name() -> None:
    one = hashlib.sha256(b"1").hexdigest()
    two = hashlib.sha256(b"2").hexdigest()
    assert merkle_root((("b", two), ("a", one))) == merkle_root(
        (("a", one), ("b", two))
    )
    with pytest.raises(ValueError, match="not hexadecimal"):
        merkle_root((("invalid", "not-a-digest"),))


def test_static_manifest_is_cached_once() -> None:
    cache = StaticArtifactCache()
    first = cache.resolve(_manifest())
    second = cache.resolve(_manifest())
    assert first == second
    assert len(cache) == 1


def test_online_and_full_batch_modes_preserve_same_root() -> None:
    predictions = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    explanations = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    cache = StaticArtifactCache()
    online = audit_batch(
        predictions,
        explanations,
        _manifest(),
        ("sample-1", "sample-2"),
        cache=cache,
        mode="online",
    )
    full = audit_batch(
        predictions,
        explanations,
        _manifest(),
        ("sample-1", "sample-2"),
        cache=cache,
        mode="full",
    )
    assert online.merkle_root == full.merkle_root
    assert online.serialized is None
    assert full.serialized
    assert set(online.timings_ms) == {
        "normalize_ms",
        "prediction_digest_ms",
        "explanation_digest_ms",
        "contract_check_ms",
        "route_build_ms",
        "cut_search_ms",
        "proof_trace_ms",
        "serialization_ms",
    }


def test_batch_rejects_duplicate_identity_and_shape_mismatch() -> None:
    cache = StaticArtifactCache()
    with pytest.raises(ValueError, match="unique"):
        audit_batch(
            np.ones((2, 1)),
            np.ones((2, 1)),
            _manifest(),
            ("x", "x"),
            cache=cache,
        )
    with pytest.raises(ValueError, match="batch sizes"):
        audit_batch(
            np.ones((2, 1)),
            np.ones((1, 1)),
            _manifest(),
            ("x", "y"),
            cache=cache,
        )

from __future__ import annotations

from scripts.q1_final.merge_real_benchmarks import _onnx_verified


def test_optional_onnx_channel_is_null_safe() -> None:
    assert _onnx_verified(None) is False
    assert _onnx_verified({"onnx": None}) is False
    assert _onnx_verified({"onnx": {"status": "not_available"}}) is False
    assert _onnx_verified({"onnx": {"status": "verified"}}) is True

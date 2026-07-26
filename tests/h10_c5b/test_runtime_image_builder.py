from __future__ import annotations

import pytest

from scripts.ch4_revision.build_h10_c5b_runtime_images import render_dockerfile


def test_runtime_compatibility_dockerfile_is_digest_and_wheel_pinned() -> None:
    base = "registry.example/project@sha256:" + "a" * 64
    wheel_sha = "b" * 64
    first = render_dockerfile(base, "numpy-1.26.4-cp311.whl", wheel_sha)
    second = render_dockerfile(base, "numpy-1.26.4-cp311.whl", wheel_sha)

    assert first == second
    assert first.startswith(f"FROM {base}\n")
    assert wheel_sha in first
    assert "--no-index --no-deps" in first
    assert "numpy.__version__ == '1.26.4'" in first
    assert "runtime-environment-v1" in first


@pytest.mark.parametrize(
    "base",
    [
        "registry.example/project:latest",
        "registry.example/project@sha256:short",
        "registry.example/project@sha256:" + "G" * 64,
    ],
)
def test_runtime_compatibility_dockerfile_rejects_unpinned_base(base: str) -> None:
    with pytest.raises(ValueError, match="pinned"):
        render_dockerfile(base, "numpy.whl", "b" * 64)


def test_runtime_compatibility_dockerfile_rejects_unsafe_wheel() -> None:
    base = "registry.example/project@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="unsafe"):
        render_dockerfile(base, "../numpy.whl", "b" * 64)

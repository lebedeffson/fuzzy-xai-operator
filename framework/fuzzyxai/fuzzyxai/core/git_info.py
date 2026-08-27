from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def get_source_commit() -> str:
    """Read the current object id without invoking a VCS command.

    This repository's execution policy forbids spawning Git, including for
    read-only provenance. A loose or packed ref can still be read as ordinary
    filesystem metadata; exported source trees honestly return ``unknown``.
    """

    try:
        metadata = find_repo_root(Path(__file__)) / ".git"
        if metadata.is_file():
            pointer = metadata.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return "unknown"
            metadata = (metadata.parent / pointer.split(":", 1)[1].strip()).resolve()
        head = (metadata / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or "unknown"
        ref_name = head.split(":", 1)[1].strip()
        loose_ref = metadata / ref_name
        if loose_ref.exists():
            return loose_ref.read_text(encoding="utf-8").strip() or "unknown"
        packed = metadata / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                object_id, _, name = line.partition(" ")
                if name == ref_name:
                    return object_id or "unknown"
        return "unknown"
    except Exception:
        return "unknown"

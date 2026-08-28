from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nrrd
import numpy as np


@dataclass(frozen=True)
class Structure:
    structure_id: int
    acronym: str
    name: str
    parent_id: int | None


def load_volumes(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    nissl, nissl_header = nrrd.read(root / "ara_nissl_25.nrrd")
    annotation, annotation_header = nrrd.read(root / "annotation_25.nrrd")
    if nissl.shape != annotation.shape:
        raise ValueError(f"Nissl/annotation shape mismatch: {nissl.shape} != {annotation.shape}")
    return np.asarray(nissl), np.asarray(annotation), nissl_header, annotation_header


def _walk(node: dict[str, Any], parent_id: int | None, result: list[Structure]) -> None:
    structure_id = int(node["id"])
    result.append(Structure(structure_id, str(node["acronym"]), str(node["name"]), parent_id))
    for child in node.get("children", []):
        _walk(child, structure_id, result)


def load_structures(path: Path) -> list[Structure]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    roots = payload.get("msg", payload)
    if isinstance(roots, dict):
        roots = [roots]
    result: list[Structure] = []
    for root in roots:
        _walk(root, None, result)
    return result


def descendants(structures: list[Structure], acronym: str) -> set[int]:
    matches = [item for item in structures if item.acronym.lower() == acronym.lower()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ontology acronym {acronym}, found {len(matches)}")
    selected, changed = {matches[0].structure_id}, True
    while changed:
        before = len(selected)
        selected.update(item.structure_id for item in structures if item.parent_id in selected)
        changed = len(selected) != before
    return selected


def patch_hash(patch: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(patch).tobytes()).hexdigest()

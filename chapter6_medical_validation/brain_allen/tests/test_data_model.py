from __future__ import annotations

import json

import numpy as np
import torch

from chapter6_medical_validation.brain_allen.src.data import descendants, load_structures, patch_hash
from chapter6_medical_validation.brain_allen.src.model import build_inception_binary


def test_ontology_hpf_descendants_are_resolved_by_acronym(tmp_path):
    payload = {"msg": [{"id": 8, "acronym": "grey", "name": "Basic cell groups", "children": [{"id": 100, "acronym": "HPF", "name": "Hippocampal formation", "children": [{"id": 101, "acronym": "CA", "name": "Ammon's horn", "children": []}]}]}]}
    path = tmp_path / "graph.json"; path.write_text(json.dumps(payload))
    structures = load_structures(path)
    assert descendants(structures, "HPF") == {100, 101}
    assert descendants(structures, "grey") == {8, 100, 101}


def test_patch_hash_is_deterministic_and_inception_output_is_binary():
    patch = np.arange(64, dtype=np.uint16).reshape(8, 8)
    assert patch_hash(patch) == patch_hash(patch.copy())
    model = build_inception_binary(pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 299, 299))
    assert output.shape == (1, 2)

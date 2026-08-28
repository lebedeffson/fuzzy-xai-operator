from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("FUZZYXAI_CH6_DATA_ROOT"), reason="real Allen data root not registered")
def test_canonical_checkpoint_replays_saved_test_classes() -> None:
    import torch

    from chapter6_medical_validation.brain_allen.scripts.run_cases import canonical_run
    from chapter6_medical_validation.brain_allen.src.model import build_inception_binary
    from chapter6_medical_validation.brain_allen.src.preprocessing import preprocess_patch

    prepared = Path(os.environ["FUZZYXAI_CH6_DATA_ROOT"]) / "brain" / "allen_ccf_25um" / "prepared"
    patches = np.load(prepared / "patches.npy", mmap_mode="r")
    metadata = json.loads((prepared / "patches.json").read_text(encoding="utf-8"))
    run_dir, run = canonical_run()
    saved = np.load(run_dir / "test_predictions.npz")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cuda", weights_only=False)
    # This frozen v1 pilot was trained under the old no-pretrained constructor
    # contract.  v2 records and tests its distinct pretrained=True contract.
    model = build_inception_binary(pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    predictions = []
    for index in saved["prepared_indices"]:
        tensor, _, _ = preprocess_patch(np.asarray(patches[int(index)]), scale=float(run["preprocessing"]["scale"]), split="test", seed=int(run["seed"]), object_index=int(index))
        with torch.no_grad():
            predictions.append(int(model(tensor[None].cuda()).argmax(dim=1).item()))
    assert predictions == np.argmax(saved["logits"], axis=1).tolist()
    assert predictions == [int(metadata[int(index)]["label"]) for index in saved["prepared_indices"]]


@pytest.mark.skipif(not os.environ.get("FUZZYXAI_CH6_DATA_ROOT"), reason="real Allen data root not registered")
def test_v2_checkpoint_replays_saved_test_classes() -> None:
    import torch

    from chapter6_medical_validation.brain_allen.scripts.run_cases import canonical_run
    from chapter6_medical_validation.brain_allen.src.model import build_inception_binary
    from chapter6_medical_validation.brain_allen.src.preprocessing import preprocess_patch

    prepared = Path(os.environ["FUZZYXAI_CH6_DATA_ROOT"]) / "brain" / "allen_ccf_25um" / "prepared_v2_confirmatory"
    patches = np.load(prepared / "patches.npy", mmap_mode="r")
    run_dir, run = canonical_run("outputs_v2_confirmatory")
    saved = np.load(run_dir / "test_predictions.npz")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cuda", weights_only=False)
    model = build_inception_binary(pretrained=True).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    predictions = []
    for index in saved["prepared_indices"]:
        tensor, _, _ = preprocess_patch(np.asarray(patches[int(index)]), scale=float(run["preprocessing"]["scale"]), split="test", seed=int(run["seed"]), object_index=int(index))
        with torch.no_grad():
            predictions.append(int(model(tensor[None].cuda()).argmax(dim=1).item()))
    assert predictions == np.argmax(saved["logits"], axis=1).tolist()

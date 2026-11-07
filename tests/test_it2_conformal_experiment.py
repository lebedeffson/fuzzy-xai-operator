"""Holdout conformal comparison exercises the original WDBC experiment pipeline."""
from __future__ import annotations

import csv
import json

from experiments.doctoral_it2_conformal import run, run_one
from fuzzyxai.it2_conformal import IT2SplitConformal


def test_train_validation_test_are_disjoint_and_conformal_is_replayable(tmp_path):
    result = run_one(17, tmp_path, n_bootstrap=8)
    assert result["training_n"] + result["calibration_n"] + result["test_n"] == 569
    assert len(set(result["split_sha256"].values())) == 3
    assert result["calibration_n"] >= 50
    assert result["status"] == "measured_on_held_out_rows"
    for name in ("it2_conformal", "type1_conformal", "it2_conformal_only"):
        metrics = result[name]
        assert metrics["n_test"] == result["test_n"]
        for key in ("empirical_marginal_coverage", "automatic_accept_fraction"):
            assert 0 <= metrics[key] <= 1
        assert 0 <= metrics["mean_set_size"] <= 2
        assert 0 <= metrics["automatic_accept_count"] <= metrics["n_test"]
        assert metrics["automatically_accepted_false_negatives"] <= metrics["automatic_accept_count"]
    for variant in ("it2", "type1"):
        evidence = json.loads((tmp_path / f"calibration_{variant}.json").read_text(encoding="utf-8"))
        replay = IT2SplitConformal.from_dict(evidence)
        assert replay.receipt_sha256 == result["calibration_receipts"][variant]
        assert len(replay.calibration_scores) == result["calibration_n"]
    with (tmp_path / "heldout_conformal.csv").open(encoding="utf-8", newline="") as handle:
        heldout = list(csv.DictReader(handle))
    assert len(heldout) == result["test_n"]
    assert len({row["row_id"] for row in heldout}) == len(heldout)
    assert all(row["it2_combined_accept"] in {"0", "1"} for row in heldout)


def test_repeated_seed_output_has_sha_manifest_and_is_descriptive(tmp_path):
    summary = run(tmp_path, seeds=(17, 42), n_bootstrap=8)
    assert summary["seeds"] == [17, 42]
    aggregate = summary["aggregates_descriptive"]["it2_conformal"]
    assert 0 <= aggregate["empirical_marginal_coverage_mean"] <= 1
    assert "std_descriptive" in " ".join(aggregate)
    assert "independent" in summary["independence_warning"]
    manifest = json.loads((tmp_path / "manifest_sha256.json").read_text(encoding="utf-8"))
    assert "seed_17/summary.json" in manifest["files"]
    assert "seed_42/heldout_conformal.csv" in manifest["files"]

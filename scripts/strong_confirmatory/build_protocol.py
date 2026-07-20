#!/usr/bin/env python3
"""Build the immutable formative protocol without opening confirmatory data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "study/strong_confirmatory"
FROZEN = (
    "e34e52fb8ae62ee1be043d6d5b26a0c9214a0572",
    "bd48a9ca3795e2665e0e6a4f1ab4f4e981774c2b",
    "1f5fd774afadde5fe03aed07eaf44f3f54967736",
)


def main() -> None:
    for commit in FROZEN:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write("frozen_claims.json", _frozen_claims())
    _write("comparator_taxonomy.json", _taxonomy())
    _write("dataset_plan.json", _datasets())
    _write("power_analysis.json", _power())
    protocol = _protocol()
    _write("protocol_v1.json", protocol)
    files = []
    for path in sorted(OUTPUT.glob("*.json")):
        if path.name == "manifest.json":
            continue
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)})
    _write(
        "manifest.json",
        {
            "schema_version": "1.0",
            "stage": "formative_development",
            "implementation_commit": _git("rev-parse", "HEAD"),
            "frozen_ancestors": list(FROZEN),
            "confirmatory_protocol_locked": False,
            "confirmatory_test_opened": False,
            "stable_release_allowed": False,
            "files": files,
        },
    )
    print(f"PASS: strong_protocol hypotheses=7 families=3 frozen={len(FROZEN)} confirmatory_opened=false")


def _frozen_claims() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "immutable_original_statuses": {
            "H3-original": "not_supported",
            "H5-P-original": "not_supported",
            "H6-general": "not_supported",
        },
        "retained_supported_results": {
            "H1": {"n_pairs": 6000, "status": "supported", "scope": "paired fidelity preservation"},
            "H2": {"n_deletions": 2000, "f1": 1.0, "status": "supported", "scope": "controlled provenance localization"},
            "H4": {"n_objects": 504260, "complexity_before": 8.0, "complexity_after": 3.137, "status": "supported"},
            "H5-S": {"n_faults": 4000, "f1": 1.0, "status": "supported", "scope": "controlled structural detection only"},
        },
        "new_hypotheses": {
            "H3-v2": "selective observer utility in a validation-defined ambiguity zone",
            "H5-A": "route validity prevents invalid automatic action",
            "H6-A": "planted-rule recovery validates the ablation method",
            "H6-B": "low-redundancy subgroup rules exceed matched controls",
            "H7": "system explanation stability with fidelity non-inferiority",
            "H8": "action stability across a frozen component grid",
            "H9": "near-linear operator-layer streaming scalability",
        },
    }


def _taxonomy() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "families": {
            "A_post_hoc_same_frozen_model": {
                "methods": ["SHAP", "LIME", "Anchors", "Integrated Gradients", "Grad-CAM", "token masking", "window masking", "modality-specific explainer", "same explainer inside FuzzyXAI"],
                "equal_reason_budget_required": True,
            },
            "B_interpretable_predictors": {
                "methods": ["GAM", "GA2M", "EBM", "RuleFit", "sparse decision tree", "rule list", "black-box predictor", "black-box predictor plus FuzzyXAI audit layer"],
                "applicable_inputs": "feature representations comparable to tabular predictors",
                "forbidden_claim": "FuzzyXAI is a better predictive model than EBM",
            },
            "C_action_policies": {
                "methods": ["confidence threshold", "calibrated confidence", "uncertainty threshold", "model disagreement", "explainer disagreement", "data-quality guardrail", "shift detector", "provenance-only guardrail", "simple OR", "weighted score", "conformal/selective baseline", "selective FuzzyXAI observer", "always accept", "always review"],
            },
        },
        "FAST": {"status": "excluded_pending_exact_article_and_implementation"},
    }


def _datasets() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "formative": {
            "tabular": ["sklearn breast cancer", "sklearn wine"],
            "image": ["sklearn digits"],
            "text": ["20 Newsgroups development subset"],
            "timeseries": ["controlled periodic-signal development fixture"],
        },
        "confirmatory": {
            "status": "not_opened",
            "requirements": ["at least two independent tabular datasets", "one independent image dataset", "one independent text dataset", "one independent time-series dataset with a frozen temporal split"],
            "exact_manifests": [],
            "lock_allowed": False,
        },
        "split_rules": ["group split for repeated subjects", "temporal split for time series", "OOF policy features", "untouched confirmatory test"],
    }


def _power() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "preregistered_planning_values_not_results",
        "alpha_familywise": 0.05,
        "power": 0.80,
        "minimum_confirmatory_units": {
            "H3-v2_per_modality": 1000,
            "H5-A_per_fault_type": 200,
            "H6-A_configurations": 48,
            "H6-B_candidates_per_dataset": 20,
            "H7_paired_replicates_per_method": 50,
            "H8_objects_per_modality": 1000,
        },
        "effect_thresholds": {"H3_relative_wrong_automatic_reduction": 0.15, "H3_coverage_gain": 0.05, "H7_stability_gain": 0.02, "H6_specific_effect": 0.01},
        "final_sample_sizes_require_confirmatory_dataset_prevalence": True,
    }


def _protocol() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "study_id": "fuzzyxai-strong-confirmatory-closure",
        "stage": "formative_development",
        "confirmatory_test_opened": False,
        "primary_endpoints": {
            "H3-v2": ["wrong automatic actions at matched coverage", "coverage at matched selective risk"],
            "H5-A": ["fault F1", "false certification", "source localization", "invalid-action recall"],
            "H6-A": ["planted-rule detection", "subgroup localization", "matched-control separation"],
            "H6-B": ["candidate effect minus median matched-control effect"],
            "H7": ["Jaccard@k", "Kendall tau", "sign agreement", "rank-biased overlap", "fidelity non-inferiority"],
            "H8": ["action agreement", "representation agreement", "risk difference"],
            "H9": ["scaling exponent", "streaming memory", "determinism"],
        },
        "statistics": ["paired bootstrap", "cluster bootstrap", "Wilcoxon", "McNemar", "permutation test", "Holm correction"],
        "reporting_required": ["effect size", "95% confidence interval", "adjusted p-value", "N", "unit of analysis"],
        "forbidden": ["positive confirmatory wording from formative data", "changing thresholds after test opening", "overwriting original negative results", "stable tag while external gates are open"],
    }


def _write(name: str, value: object) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()

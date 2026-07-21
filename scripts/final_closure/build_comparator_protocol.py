#!/usr/bin/env python3
"""Freeze comparator taxonomy and resolve the ambiguous FAST baseline."""

from __future__ import annotations

from common import STUDY, write


def main() -> None:
    write(
        STUDY / "comparator_taxonomy.json",
        {
            "schema_version": "1.0",
            "post_hoc_explainers_same_frozen_model": [
                "SHAP",
                "LIME",
                "Anchors",
                "Integrated Gradients",
                "Grad-CAM",
                "token masking",
                "window masking",
                "component occlusion inside FuzzyXAI",
            ],
            "interpretable_predictors_tabular_only": [
                "GAM",
                "GA2M",
                "EBM",
                "RuleFit",
                "sparse decision tree",
                "rule list",
                "FXAM if a pinned reproducible implementation becomes available",
            ],
            "action_policies": [
                "always accept",
                "always review",
                "raw confidence threshold",
                "calibrated confidence threshold",
                "uncertainty threshold",
                "model disagreement",
                "explainer disagreement",
                "provenance completeness",
                "data-quality guardrail",
                "simple OR guardrail",
                "weighted linear score",
                "predictive-risk P0",
                "conformal/selective",
                "full FuzzyXAI P1",
            ],
            "forbidden_comparisons": [
                "treating EBM, GAM, RuleFit, rule lists or FXAM as local post-hoc explainers",
                "claiming that FuzzyXAI is a better predictive model than a glass-box predictor",
                "running tabular glass-box predictors on raw images, raw text or raw time-series tensors",
            ],
            "benchmark_status": "protocol_frozen_measurements_pending",
        },
    )
    (STUDY / "baseline_resolution_FAST.md").write_text(
        "# FAST baseline resolution\n\n"
        "The acronym `FAST` is excluded from confirmatory tables because the reviewer reference does not identify a unique method.\n\n"
        "The likely candidate is **FXAM (Fast and eXplainable Additive Model)** from *A Unified and Fast Interpretable Model for Predictive Analytics*, arXiv:2111.08255. The paper is pinned as the semantic reference, but no official, versioned implementation with a reproducible package contract was identified for this cycle.\n\n"
        "Therefore FXAM is recorded as `excluded_no_pinned_reproducible_implementation`; it is not silently replaced by another additive model and the name `FAST` is not used in result tables.\n",
        encoding="utf-8",
    )
    write(
        STUDY / "comparator_resolution.json",
        {
            "FAST": "excluded_ambiguous_identifier",
            "FXAM": "excluded_no_pinned_reproducible_implementation",
            "FXAM_reference": "https://arxiv.org/abs/2111.08255",
            "search_cutoff": "2026-07-21",
            "result_claim_allowed": False,
        },
    )
    print("PASS: final_comparator_protocol families=3 FAST=excluded FXAM=excluded_unpinned")


if __name__ == "__main__":
    main()

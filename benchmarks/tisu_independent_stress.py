"""Secondary post-hoc stress suite for typed F0 composition.

The mutation set was authored after the initial 20-scenario protocol.
It is intentionally NOT a blinded, independently adjudicated experiment,
nor evidence of field performance or clinical effectiveness.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.tisu_composition_validation import BETA, POLICY, build_pair, term_specs
from fuzzyxai.core.composition import compose
from fuzzyxai.core.composition_checked import MembershipSpec, compose_checked, result_status
from fuzzyxai.core.explanation_object import Rule
from fuzzyxai.hierarchy.f0 import F0

# Fixed after the initial RF/split-42 experiment. This reuses the dataset
# but changes the model family, held-out split, and intervention generator.
SPLIT_SEED = 2026
FAULT_SEED = 82719
CHALLENGES = (
    ("unchanged_pair", "ACCEPT"),
    ("complementary_probability", "ACCEPT"),
    ("small_probability_drift", "ACCEPT"),
    ("null_trace_source", "UNKNOWN"),
    ("blank_trace_checksum", "UNKNOWN"),
    ("changed_rule_conclusion", "REJECT"),
    ("unregistered_model_revision", "REJECT"),
    ("unverifiable_f0_callback", "UNKNOWN"),
    ("random_subgrid_membership_bump", "REJECT"),
    ("missing_fuzzy_coordinate", "UNKNOWN"),
    ("corrupted_external_reference", "REJECT"),
    ("nonfinite_explanation_uncertainty", "REJECT"),
)


def intervene(kind, left, right, origin, p, rng):
    """Apply an independently specified contract change to the right module."""
    if kind == "unchanged_pair":
        pass
    elif kind == "complementary_probability":
        right.metadata["target_event"] = "benign"
        right.metadata["probability"] = 1.0 - p
        right.metadata["mapping"] = "complement"
    elif kind == "small_probability_drift":
        right.metadata["probability"] = p + (0.007 if p <= 0.95 else -0.007)
    elif kind == "null_trace_source":
        right.trace = replace(right.trace, source=None)
    elif kind == "blank_trace_checksum":
        right.trace = replace(right.trace, checksum="   ")
    elif kind == "changed_rule_conclusion":
        right.rules = [Rule("risk_rule", {"risk": "high"}, "different_action")]
    elif kind == "unregistered_model_revision":
        right.metadata["model_version"] = "unregistered-v" + str(rng.randint(2, 5000))
    elif kind == "unverifiable_f0_callback":
        declared_spec = term_specs()["high"]
        right.representation = F0(lambda x: declared_spec.value(x), "high-risk")
    elif kind == "random_subgrid_membership_bump":
        # Perturb a random interval far from the ordinary high-risk support:
        # max deviation 0.35, tolerance 0.08. The curve is still a valid PWL F0.
        center = rng.uniform(0.021, 0.079)
        knots = (
            (0.0, 0.0), (center - 0.001, 0.0), (center, 0.35),
            (center + 0.001, 0.0), (0.5, 0.0), (0.75, 1.0), (1.0, 1.0),
        )
        changed = MembershipSpec(knots)
        # copy_with() shallow-copies metadata: never mutate the shared nested
        # term-spec dictionary on the left/reference explanation.
        right.metadata["term_specs"] = {
            **right.metadata["term_specs"], "high": changed.to_json(),
        }
        right.representation = F0(changed.value, "high-risk")
        right.activations["risk_rule"] = changed.value(p)
    elif kind == "missing_fuzzy_coordinate":
        del right.metadata["fuzzy_event"]
    elif kind == "corrupted_external_reference":
        shifted = p + (0.25 if p <= 0.5 else -0.25)
        origin = replace(origin, probability=shifted)
    elif kind == "nonfinite_explanation_uncertainty":
        right.uncertainty = float("nan")
    else:
        raise ValueError(kind)
    return left, right, origin


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    y = (data.target == 0).astype(int)  # class 1 is malignant
    x_train, x_test, y_train, _ = train_test_split(
        data.data, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=SPLIT_SEED),
    ).fit(x_train, y_train)
    p_malignant = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    rng = random.Random(FAULT_SEED)
    records = []
    totals = Counter()
    base_totals = Counter()
    for i, p in enumerate(p_malignant):
        p = float(p)
        for kind, expected in CHALLENGES:
            left, right, origin = build_pair(
                i, p, model_id="logistic_bcw_stress",
                model_version="logreg_scaled_split2026",
            )
            left, right, origin = intervene(kind, left, right, origin, p, rng)
            actual = result_status(compose_checked(left, right, BETA, origin=origin, policy=POLICY))
            baseline = result_status(compose(left, right, BETA))
            records.append((i, kind, expected, actual, baseline))
            totals[(kind, actual)] += 1
            base_totals[(kind, baseline)] += 1
    disagreements = [
        {"row": i, "case": kind, "expected": truth, "observed": observed}
        for i, kind, truth, observed, _ in records if truth != observed
    ]
    expected_accepted = sum(truth == "ACCEPT" for _, _, truth, _, _ in records)
    correct_baseline = sum(truth == baseline for _, _, truth, _, baseline in records)
    result = {
        "study": "POST_HOC_secondary_randomized_contract_stress_not_blinded",
        "model": "StandardScaler+LogisticRegression",
        "dataset": "Wisconsin Diagnostic Breast Cancer (same source, alternate split)",
        "split_seed": SPLIT_SEED,
        "fault_seed": FAULT_SEED,
        "n_held_out": len(p_malignant),
        "n_case_types": len(CHALLENGES),
        "n_checks": len(records),
        "n_expected_accept": expected_accepted,
        "checked_correct": len(records) - len(disagreements),
        "upstream_correct": correct_baseline,
        "mismatch_examples": disagreements[:10],
        "truth_by_case": dict(CHALLENGES),
        "observed_by_case": {
            name: {label: totals[(name, label)] for label in ("ACCEPT", "REJECT", "UNKNOWN")}
            for name, _ in CHALLENGES
        },
        "scope_note": "Synthetic post-hoc checks on correlated repeated cases; external origin unverified.",
    }
    (out_dir / "secondary_stress.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if disagreements:
        raise AssertionError(f"{len(disagreements)} independent stress-case mismatches: {disagreements[:3]}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tisu-secondary-stress"))
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))

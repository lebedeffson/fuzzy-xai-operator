"""Train-only bootstrap uncertainty for a genuinely operative IT2 fuzzy-logit.

The existing Type-1 native fuzzy logistic model learns coefficients from fuzzy
memberships. This extension learns *uncertainty around membership-function
knots* using the training split only, and performs interval-valued inference
at test time. It reports the exact prediction interval induced by the IT2 FOU
for the additive paired-rule logit (not a confidence interval on labels).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.interval_type2 import (
    IT2FeatureRule, IT2FuzzyLogit, IT2Shoulder, content_sha256, make_it2_receipt,
    robust_interval_decision, verify_it2_receipt,
)
from .doctoral_native_fuzzy_logistic import NativeFuzzyLogistic


class NativeIntervalType2Logistic:
    """Train a Type-1 fuzzy logistic core; bootstrap its uncertain T2 shoulders.

    This is IT2 inference with coefficients *learned on nominal Type-1 terms*;
    it is not a bespoke end-to-end optimized Type-2 neural training algorithm.
    The IT2 range changes the model's point output and abstention behavior.
    """

    def __init__(self, *, n_features: int = 5, n_bootstrap: int = 64,
                 seed: int = 42, band: tuple[float, float] = (0.05, 0.95),
                 max_width: float = 0.15):
        if n_bootstrap < 4 or not (0 < band[0] < band[1] < 1):
            raise ValueError("require at least four bootstrap resamples and ordered interior percentiles")
        if not 0 <= max_width <= 1:
            raise ValueError("max_width must be within [0,1]")
        self.n_features, self.n_bootstrap, self.seed = n_features, n_bootstrap, seed
        self.band, self.max_width = band, max_width
        self.type1 = NativeFuzzyLogistic(n_features=n_features)
        self.it2: IT2FuzzyLogit | None = None
        self.train_sha256 = ""

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> NativeIntervalType2Logistic:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.type1.fit(x, y, feature_names=feature_names)
        self.train_sha256 = content_sha256({"X": x.tolist(), "y": y.tolist()})
        rng = np.random.default_rng(self.seed)
        bootstrap_indices = rng.integers(0, len(x), (self.n_bootstrap, len(x)))
        # Same bootstrap draws for all rules, so source uncertainty can be
        # shared between features. The inference rectangle is conservative
        # when these empirical parameter changes are correlated.
        features: list[IT2FeatureRule] = []
        for feature in self.type1.features:
            vals = x[:, feature.index]
            boundaries = np.quantile(vals[bootstrap_indices], [0.02, 0.98], axis=1).T
            # `np.quantile` over (n_bootstrap, n_train) with axis=1 returns
            # shape (2,n_bootstrap); transpose -> bootstrap x low/high.
            lefts, rights = boundaries[:, 0], boundaries[:, 1]
            left_min = float(min(np.quantile(lefts, self.band[0]), feature.low))
            left_max = float(max(np.quantile(lefts, self.band[1]), feature.low))
            right_min = float(min(np.quantile(rights, self.band[0]), feature.high))
            right_max = float(max(np.quantile(rights, self.band[1]), feature.high))
            if left_max >= right_min:
                raise ValueError(f"not identifiable IT2 shoulder for {feature.name}: parameter bands cross")
            features.append(IT2FeatureRule(
                feature_index=feature.index, feature_name=feature.name,
                low_coefficient=feature.low_consequent, high_coefficient=feature.high_consequent,
                membership=IT2Shoulder(left_min, left_max, right_min, right_max,
                                       feature.low, feature.high),
            ))
        self.it2 = IT2FuzzyLogit(
            intercept=self.type1.intercept,
            n_input_features=x.shape[1],
            features=tuple(features),
            source_fit=f"training_only_bootstrap_n={self.n_bootstrap}_percentiles={self.band}",
        )
        return self

    @property
    def manifest(self) -> dict[str, Any]:
        if self.it2 is None:
            raise RuntimeError("fit model before inference")
        return self.it2.to_dict()

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.it2 is None:
            raise RuntimeError("fit model first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        lower, upper = [], []
        for row in x:
            out = self.it2.infer(row)
            lower.append(out["probability_lower"])
            upper.append(out["probability_upper"])
        return np.asarray(lower), np.asarray(upper)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_interval(x)
        p = (lo + hi) / 2.0
        return np.column_stack([1.0 - p, p])

    def decision(self, row: np.ndarray) -> dict[str, Any]:
        if self.it2 is None:
            raise RuntimeError("fit model first")
        return robust_interval_decision(self.it2.infer(row), max_width=self.max_width)


def _binary_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    pred = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_malignant": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def _selective_metrics(y: np.ndarray, p: np.ndarray, accepted: np.ndarray) -> dict[str, float | int | None]:
    n = len(y)
    count = int(accepted.sum())
    return {
        "n": n,
        "accepted": count,
        "coverage": count / n,
        "selective_error": (float(np.mean((p[accepted] >= .5) != y[accepted])) if count else None),
        "selective_malignant_false_negative_count": int(np.sum((y[accepted] == 1) & (p[accepted] < .5))),
    }


def run_one(seed: int, out_dir: Path, *, n_features: int = 5,
            n_bootstrap: int = 64, max_width: float = 0.15) -> dict[str, Any]:
    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=float)
    y = (np.asarray(data.target) == 0).astype(int)
    identities = np.arange(len(y))
    X_tr, X_other, y_tr, y_other, ids_tr, ids_other = train_test_split(
        X, y, identities, test_size=0.4, stratify=y, random_state=seed,
    )
    X_val, X_te, y_val, y_te, ids_val, ids_te = train_test_split(
        X_other, y_other, ids_other, test_size=0.5, stratify=y_other, random_state=seed + 1,
    )
    model = NativeIntervalType2Logistic(n_features=n_features, n_bootstrap=n_bootstrap,
                                       seed=seed, max_width=max_width)
    model.fit(X_tr, y_tr, feature_names=list(data.feature_names))
    p_type1 = model.type1.predict_proba(X_te)[:, 1]
    p_it2 = model.predict_proba(X_te)[:, 1]
    lo, hi = model.predict_interval(X_te)
    val_p1 = model.type1.predict_proba(X_val)[:, 1]
    val_it2_accept = np.asarray([model.decision(row)["accepted"] for row in X_val])
    target_coverage = float(val_it2_accept.mean())
    # Matched-coverage Type-1 abstention threshold is chosen using validation
    # ONLY. Never choose a threshold from held-out test labels or results.
    val_margins = np.abs(val_p1 - 0.5)
    rank = min(len(val_margins) - 1, max(0, int(round((1 - target_coverage) * len(val_margins)))))
    margin_threshold = float(np.sort(val_margins)[rank]) if target_coverage < 1 else -1.0
    baseline_accept = np.abs(p_type1 - 0.5) > margin_threshold
    it2_actions = [model.decision(row) for row in X_te]
    it2_accept = np.asarray([action["accepted"] for action in it2_actions], dtype=bool)
    widths = hi - lo
    # A Type-1 zero-uncertainty ablation: same learned weights, no IT2 FOU.
    # Distinct predictors are evaluated on exactly the same test objects.
    test_outputs = []
    for i, row in enumerate(X_te):
        test_outputs.append({
            "dataset_row_id": int(ids_te[i]), "y_malignant": int(y_te[i]),
            "p_type1": float(p_type1[i]), "p_it2_midpoint": float(p_it2[i]),
            "p_it2_lower": float(lo[i]), "p_it2_upper": float(hi[i]),
            "it2_width": float(widths[i]), "it2_action": it2_actions[i]["action"],
            "it2_accept": bool(it2_accept[i]), "t1_matched_accept": bool(baseline_accept[i]),
        })
    if np.any(lo > p_type1 + 1e-10) or np.any(hi < p_type1 - 1e-10):
        raise AssertionError("nominal Type-1 probability must be contained in IT2 interval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "heldout_it2.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_outputs[0]))
        writer.writeheader()
        writer.writerows(test_outputs)
    dataset_version = f"sklearn:breast_cancer:seed={seed}:train={len(X_tr)}:val={len(X_val)}:test={len(X_te)}"
    receipt = make_it2_receipt(model.it2, X_te[0].tolist(), source_id=f"wdbc:{int(ids_te[0])}",
                              dataset_version=dataset_version, max_width=max_width)
    verification = verify_it2_receipt(receipt, trusted_model_sha256=content_sha256(model.manifest))
    if not verification["valid"]:
        raise AssertionError(verification["errors"])
    (out_dir / "it2_model_manifest.json").write_text(json.dumps(model.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sample_type2_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    widths_summary = {"mean": float(np.mean(widths)), "p90": float(np.quantile(widths, .9)),
                      "max": float(np.max(widths)), "positive_width_cases": int(np.sum(widths > 1e-12))}
    report = {
        "study": "exploratory_train_only_IT2_antecedent_bootstrap",
        "seed": seed,
        "data": "sklearn.datasets.load_breast_cancer",
        "n_train": len(X_tr), "n_validation": len(X_val), "n_test": len(X_te),
        "row_split_sha256": {"train": content_sha256(ids_tr.tolist()), "validation": content_sha256(ids_val.tolist()), "test": content_sha256(ids_te.tolist())},
        "train_matrix_sha256": model.train_sha256,
        "bootstrap_n": n_bootstrap, "bootstrap_knot_quantiles": [0.02, 0.98],
        "empirical_band_percentiles": list(model.band),
        "model_manifest_sha256": content_sha256(model.manifest),
        "metrics": {"type1_native": _binary_metrics(y_te, p_type1), "interval_type2_midpoint": _binary_metrics(y_te, p_it2)},
        "interval_width": widths_summary,
        "abstention": {
            "it2": _selective_metrics(y_te, p_it2, it2_accept),
            "type1_matched_validation_coverage": _selective_metrics(y_te, p_type1, baseline_accept),
            "validation_it2_coverage": target_coverage,
            "validation_matched_type1_margin_threshold": margin_threshold,
            "it2_actions": {action: sum(v["action"] == action for v in it2_actions) for action in ("accept", "request_more_data", "defer_to_human", "block")},
        },
        "certificate_replayed": verification["valid"],
        "scope": "parameter_induced_interval_not_statistical_prediction_CI",
        "limitations": [
            "IT2 membership parameters estimated on training bootstrap; logistic consequents trained on nominal Type-1 degrees",
            "No model is trained end-to-end to optimize the interval Type-2 loss",
            "No clinical claim; the WDBC dataset is a public diagnostic tabular benchmark",
            "The FOU is a conservative rectangular enclosure of selected empirical parameter bands",
            "IT2 interval is not a calibrated interval for the true clinical outcome",
            "Repeated splits share subjects; any aggregation is descriptive, not independent replication",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(root: Path, *, seeds: tuple[int, ...] = (17, 42, 73, 123, 2026),
        n_features: int = 5, n_bootstrap: int = 64, max_width: float = 0.15) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seeds required")
    root.mkdir(parents=True, exist_ok=True)
    reports = [run_one(seed, root / f"seed_{seed}", n_features=n_features,
                       n_bootstrap=n_bootstrap, max_width=max_width) for seed in seeds]
    aggregate = {}
    for variant in ("type1_native", "interval_type2_midpoint"):
        aggregate[variant] = {}
        for metric in ("accuracy", "f1_malignant", "roc_auc", "brier"):
            values = [float(rep["metrics"][variant][metric]) for rep in reports]
            aggregate[variant][metric] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else None,
                                          "min": min(values), "max": max(values)}
    summary = {
        "schema_version": "fuzzyxai-it2-wdbc-exploratory-v1",
        "status": "computed",
        "seed_list": list(seeds),
        "n_bootstrap_per_split": n_bootstrap,
        "max_width": max_width,
        "aggregate_descriptive": aggregate,
        "abstention_per_split": [{"seed": rep["seed"], **rep["abstention"]} for rep in reports],
        "width_per_split": [{"seed": rep["seed"], **rep["interval_width"]} for rep in reports],
        "limitations": reports[0]["limitations"],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        first = reports[0]["seed"]
        manifest = json.loads((root / f"seed_{first}" / "it2_model_manifest.json").read_text(encoding="utf-8"))
        f = manifest["features"][0]
        from fuzzyxai.interval_type2 import IT2Shoulder
        shoulder = IT2Shoulder(**f["membership"])
        xs = np.linspace(shoulder.left_min - .1 * (shoulder.right_max - shoulder.left_min),
                         shoulder.right_max + .1 * (shoulder.right_max - shoulder.left_min), 240)
        lower = [shoulder.high(float(x)).lower for x in xs]
        upper = [shoulder.high(float(x)).upper for x in xs]
        point = [shoulder.nominal_high(float(x)) for x in xs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(xs, lower, upper, facecolor="0.83", label="IT2 footprint of uncertainty")
        ax.plot(xs, lower, color="black", linestyle="--", label="LMF")
        ax.plot(xs, upper, color="black", linestyle="-", label="UMF")
        ax.plot(xs, point, color="black", linestyle=":", label="Nominal T1")
        ax.set(xlabel=f["feature_name"] + " (train-derived)", ylabel="High-term membership", ylim=(-.02, 1.02),
               title="Interval type-2 membership (empirical FOU, no coverage claim)")
        ax.grid(axis="y", linestyle=":", alpha=.35)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "it2_membership_FOU.png", dpi=180)
        fig.savefig(root / "it2_membership_FOU.pdf")
        plt.close(fig)
    except ImportError:
        pass
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest_sha256.json"}
    (root / "manifest_sha256.json").write_text(json.dumps({"schema": "sha256-manifest-v1", "files": hashes}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/chapter5/interval_type2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73, 123, 2026])
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--max-width", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seeds=tuple(args.seeds), n_features=args.features,
                         n_bootstrap=args.bootstrap, max_width=args.max_width), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

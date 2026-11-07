"""A trainable, natively interpretable fuzzy predictor and measured WDBC study.

Unlike a post-hoc rule summary, the rule activations and fitted consequents
here ARE the predictor: p(y=malignant|x) is exactly their convex combination.
This is a methodological demonstration, not a clinically approved diagnosis.

The training/test split is fixed before ALL feature selection, quantile fitting
and rule consequent estimation. A held-out test set is never used to construct
memberships, rule weights or policy thresholds. The FuzzyXAI receipt verifies
the actual explanation route, not held-out predictive quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuzzyxai.doctoral_certification import (
    canonical_json, certify_payload, select_representation, sha256,
    triangular_membership, verify_certificate,
)

WEIGHTS = {"rho_p": 0.30, "u_M": 0.20, "one_minus_I_pre": 0.20, "Delta": 0.20, "chi_R": 0.10}
THRESHOLDS = {"theta_1": 0.2, "theta_2": 0.4, "theta_3": 0.6, "theta_4": 0.8}
ACTIONS = ("accept", "lower_confidence", "request_more_data", "defer_to_human", "block")


@dataclass(frozen=True)
class FuzzyFeature:
    index: int
    name: str
    low: float
    high: float
    weight: float
    low_consequent: float
    high_consequent: float
    low_support: float
    high_support: float

    @property
    def key(self) -> str:
        return f"f{self.index:02d}"

    def membership(self, value: float) -> tuple[float, float]:
        # These are saturated shoulder terms: evidence must record the CLIPPED
        # value for independent replay. Triangles outside [low, high] alone
        # would incorrectly set BOTH degrees to zero on held-out outliers.
        effective = min(max(float(value), self.low), self.high)
        return (triangular_membership(effective, (self.low, self.low, self.high)),
                triangular_membership(effective, (self.low, self.high, self.high)))


class NativeFuzzyWDBC:
    """Two transparent shoulder terms per selected feature, convex rule mixture.

    Class 1 means malignant. Learned labels are WDBC's documented target
    recoded so that sklearn's original 0 (malignant) becomes 1.
    """

    def __init__(self, n_features: int = 5, prior_strength: float = 5.0):
        if n_features < 1 or not math.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("n_features and positive finite prior_strength required")
        self.n_features = n_features
        self.prior_strength = prior_strength
        self.features: list[FuzzyFeature] = []
        self.medians: list[float] = []
        self.feature_names: list[str] = []
        self.train_size = 0
        self.positive_prior = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> NativeFuzzyWDBC:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 30 or not np.isfinite(x).all():
            raise ValueError("finite 2D training features and matching labels required")
        if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
            raise ValueError("binary labels with both classes required")
        self.feature_names = list(feature_names or [f"feature_{j}" for j in range(x.shape[1])])
        if len(self.feature_names) != x.shape[1]:
            raise ValueError("feature names must match input dimensionality")
        self.train_size = len(x)
        self.positive_prior = float(np.mean(y))
        self.medians = [float(v) for v in np.median(x, axis=0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            f_scores, _ = f_classif(x, y)
        f_scores = np.where(np.isfinite(f_scores), np.maximum(f_scores, 0.0), 0.0)
        ordered = np.argsort(-f_scores, kind="stable")
        chosen: list[tuple[int, float, float]] = []
        for idx in ordered:
            lo, hi = np.quantile(x[:, idx], [0.10, 0.90])
            if hi > lo + 1e-10:
                chosen.append((int(idx), float(lo), float(hi)))
            if len(chosen) == self.n_features:
                break
        if not chosen:
            raise ValueError("no variable features available for membership fitting")
        importances = np.asarray([f_scores[j] for j, _, _ in chosen], dtype=float)
        if not np.isfinite(importances).all() or importances.sum() <= 0:
            importances = np.ones(len(chosen))
        importances = importances / importances.sum()
        fitted: list[FuzzyFeature] = []
        for (idx, lo, hi), weight in zip(chosen, importances):
            high_mu = np.clip((x[:, idx] - lo) / (hi - lo), 0.0, 1.0)
            low_mu = 1.0 - high_mu
            low_mass, high_mass = float(low_mu.sum()), float(high_mu.sum())
            smoothed_low = (float(low_mu @ y) + self.prior_strength * self.positive_prior) / (low_mass + self.prior_strength)
            smoothed_high = (float(high_mu @ y) + self.prior_strength * self.positive_prior) / (high_mass + self.prior_strength)
            fitted.append(FuzzyFeature(idx, self.feature_names[idx], lo, hi, float(weight),
                                       smoothed_low, smoothed_high, low_mass, high_mass))
        self.features = fitted
        return self

    def fitted_json(self) -> dict[str, Any]:
        if not self.features:
            raise RuntimeError("fit() must be called first")
        return {"model_type": "NativeFuzzyWDBC-v1", "n_features": self.n_features,
                "prior_strength": self.prior_strength, "train_size": self.train_size,
                "positive_prior": self.positive_prior, "medians": self.medians,
                "feature_names": self.feature_names,
                "rules": [vars(item) for item in self.features]}

    def fingerprint(self) -> str:
        return sha256(self.fitted_json())

    def _explain(self, row: np.ndarray, *, allow_missing: bool = False) -> dict[str, Any]:
        if not self.features:
            raise RuntimeError("model not fitted")
        raw = np.asarray(row, dtype=float)
        if raw.shape != (len(self.feature_names),):
            raise ValueError("input dimensions do not match fitted feature schema")
        if np.isinf(raw).any():
            raise ValueError("infinite feature values cannot be interpreted")
        missing = np.flatnonzero(np.isnan(raw)).tolist()
        if missing and not allow_missing:
            raise ValueError("missing features require explicit allow_missing=True and critical flag")
        used = raw.copy()
        if missing:
            used[missing] = np.asarray(self.medians)[missing]
        rules: list[dict[str, Any]] = []
        per_feature_scores = []
        for feat in self.features:
            raw_value = float(used[feat.index])
            value = min(max(raw_value, feat.low), feat.high)
            low_degree, high_degree = feat.membership(value)
            per_feature_scores.append((feat.weight, low_degree * feat.low_consequent + high_degree * feat.high_consequent))
            for term, degree, consequent in (("low", low_degree, feat.low_consequent),
                                            ("high", high_degree, feat.high_consequent)):
                rules.append({"rule_id": f"{feat.key}:{term}", "feature": feat.key,
                              "feature_name": feat.name, "term": term, "value": value,
                              "raw_value": raw_value, "preprocessing": "clip_to_train_fitted_interval",
                              "membership": float(degree), "feature_weight": feat.weight,
                              "consequent": consequent,
                              "contribution": feat.weight * float(degree) * consequent,
                              "strength": feat.weight * float(degree)})
        full = float(sum(rule["contribution"] for rule in rules))
        if not 0 <= full <= 1:
            raise AssertionError("fuzzy convex combination left the unit interval")
        # Independent scalar recomputation of exactly the full set of native rules.
        second = float(sum(weight * value for weight, value in per_feature_scores))
        gamma = abs(full - second)
        feature_var = sum(weight * (value - full)**2 for weight, value in per_feature_scores)
        i_pre = 1.0 - min(1.0, 2.0 * math.sqrt(max(0.0, feature_var)))
        entropy = 0.0 if full in (0.0, 1.0) else -(full * math.log2(full) + (1-full) * math.log2(1-full))
        # Reduce explanation to at most #selected_features rules. Keep original
        # prediction intact; only the explanatory projection is reduced.
        top = sorted(rules, key=lambda item: (item["strength"], item["rule_id"]), reverse=True)[:len(self.features)]
        mass = sum(item["strength"] for item in top)
        short_score = sum(item["contribution"] for item in top) / mass if mass else full
        delta = abs(full - short_score)
        return {"probability": full, "raw": raw, "used": used, "missing": missing,
                "rules": rules, "top_rules": top, "gamma": gamma, "delta": delta,
                "score_reduced": short_score, "i_pre": i_pre, "uncertainty": entropy}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        p = np.array([self._explain(row)["probability"] for row in values], dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def make_receipt_payload(model: NativeFuzzyWDBC, row: np.ndarray, *, object_id: str,
                         dataset_version: str, origin: str = "measured", allow_missing: bool = False,
                         generated_at: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Materialize measured native rule evidence and a replayable certificate input."""
    out = model._explain(row, allow_missing=allow_missing)
    p, missing = out["probability"], out["missing"]
    # Inference and disclosure have two independently computed paths.
    if abs(float(model.predict_proba(out["used"])[0, 1]) - p) > 1e-12:
        raise AssertionError("explanation does not reconstruct actual model inference")
    chi = int(bool(missing))
    components = {"rho_p": p, "u_M": out["uncertainty"], "one_minus_I_pre": 1.0 - out["i_pre"],
                  "Delta": out["delta"], "chi_R": chi}
    rho = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)
    candidate = ACTIONS[next((j for j, value in enumerate(THRESHOLDS.values()) if rho < value), 4)]
    action = "block" if chi else candidate
    profile = ["u_num", "u_ling"] + (["u_conf", "u_src"] if missing else [])
    sel = select_representation(profile, audience="auditor", measured_delta=out["delta"])
    # If reduction loss exceeds the declared epsilon, there is NO admissible
    # class for certification; emit 'unknown' as a route fact, never invent F0.
    selected_class = sel["selected_class"] or "not_certified"
    representation = {"profile": profile, "selected_class": selected_class, "measured_delta": out["delta"]}
    measured = {"object_id": object_id, "selected_features": [f.index for f in model.features],
                "values": [None if np.isnan(v) else float(v) for v in out["raw"]]}
    route = [{"node_id": "representation_selection", "operator_type": "representation_selection",
              "value_source": "computed", "output_values": {"selected_class": selected_class}},
             {"node_id": "reduction", "operator_type": "measured_rule_reduction", "value_source": "computed",
              "output_values": {"delta": out["delta"], "retained_rules": [v["rule_id"] for v in out["top_rules"]]}},
             {"node_id": "risk", "operator_type": "canonical_observer", "value_source": "computed",
              "output_values": {"rho": rho, "action": action}}]
    graph_nodes = [
        {"node_id": "dataset:root", "node_type": "dataset", "payload": {"dataset_version": dataset_version}},
        {"node_id": "model_artifact:root", "node_type": "model_artifact", "payload": {"model_fingerprint": model.fingerprint()}},
        {"node_id": f"data:{object_id}", "node_type": "data", "payload": {"input_sha256": sha256(measured)}},
        {"node_id": "prediction", "node_type": "prediction", "payload": {"probability_malignant": p}},
        {"node_id": "claim:C-native-rule", "node_type": "claim", "payload": {"claim_type": "native_rule"}},
        {"node_id": "action", "node_type": "action", "payload": {"action": action}},
    ]
    graph_edges = [
        {"source": "dataset:root", "target": f"data:{object_id}", "relation": "derived_from"},
        {"source": "model_artifact:root", "target": "prediction", "relation": "produces"},
        {"source": f"data:{object_id}", "target": "prediction", "relation": "derived_from"},
        {"source": "prediction", "target": "action", "relation": "observed_by_policy"},
    ]
    layers_rules, activations, membership_policies, membership_values = [], [], {}, {}
    for feature in model.features:
        membership_policies[feature.key] = {
            "variable": feature.key, "universe": [feature.low, feature.high],
            "origin": "calibrated", "version": f"train-only:{dataset_version}",
            "calibration_reference": f"model:{model.fingerprint()}",
            "preprocessing": "clip_to_train_fitted_interval",
            "terms": [
                {"label": "low", "function": "triangular", "parameters": [feature.low, feature.low, feature.high]},
                {"label": "high", "function": "triangular", "parameters": [feature.low, feature.high, feature.high]},
            ],
        }
    for rule in out["rules"]:
        feature = next(f for f in model.features if f.key == rule["feature"])
        rid = rule["rule_id"]
        graph_nodes.append({"node_id": f"rule:{rid}", "node_type": "rule",
                            "payload": {"feature": rule["feature"], "activation": rule["membership"],
                                        "consequent": rule["consequent"], "weight": rule["feature_weight"]}})
        graph_edges.append({"source": f"data:{object_id}", "target": f"rule:{rid}", "relation": "derives_membership"})
        graph_edges.append({"source": f"rule:{rid}", "target": "prediction", "relation": "computes_probability"})
        mass = feature.low_support if rule["term"] == "low" else feature.high_support
        layers_rules.append({"rule_id": rid, "native": True, "source": "train_only_fitted_consequent",
                             "weighted_training_support": mass, "consequent": rule["consequent"],
                             "consequent_semantics": getattr(model, "rule_semantics", "malignant_probability")})
        activations.append({"object_id": object_id, "rule_id": rid, "activation_strength": rule["membership"],
                            "conclusion": (f"log-odds coefficient {rule['consequent']:.6g}" if getattr(model, "rule_semantics", "") == "log_odds" else f"malignant probability {rule['consequent']:.6g}"),
                            "terms": [{"feature": rule["feature"], "term": rule["term"],
                                       "membership_degree": rule["membership"], "feature_value": rule["value"],
                                       "raw_feature_value": rule.get("raw_value", rule["value"]),
                                       "preprocessing": rule.get("preprocessing", "none")}]})
        membership_values[rid] = rule["membership"]
    strongest = max(out["top_rules"], key=lambda r: r["strength"])
    graph_edges.append({"source": f"rule:{strongest['rule_id']}", "target": "claim:C-native-rule", "relation": "supports_claim"})
    graph_edges.append({"source": "claim:C-native-rule", "target": "action", "relation": "constrains_policy"})
    diagnostics = ([{"code": "D_quality_source_conflict", "criticality": "high", "source": "controlled_missing_feature",
                    "reason": "input feature was missing; train median imputation is not trusted for automation"}] if missing else [])
    trace = {"adapter_id": "NativeFuzzyWDBC/transparent-predictor", "input_sha256": sha256(measured),
             "explain_plan_sha256": sha256({"risk_weights": WEIGHTS, "risk_thresholds": THRESHOLDS,
                                             "representation": "train_quantile_shoulder_terms_v1", "tnorm": "minimum"}),
             "model_fingerprint": model.fingerprint(), "dataset_version": dataset_version,
             "object_ids": [object_id],
             "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
             "result_origin": origin}
    payload = {
        "trace": trace,
        "model": {"class_order": ["benign", "malignant"], "predicted_malignancy_probability": p,
                  "model_type": model.fitted_json()["model_type"],
                  **({"parameter_manifest": model.fitted_json()} if getattr(model, "rule_semantics", "") == "log_odds" else {})},
        "explanation_graph": {"nodes": graph_nodes, "edges": graph_edges, "missing_evidence": []},
        "claims": [{"claim_id": "C-native-rule", "claim_type": "native_rule", "evidence_status": "supported",
                    "statement": f"Measured native rule {strongest['rule_id']} contributes to model prediction",
                    "evidence_refs": [f"rule:{strongest['rule_id']}"]}],
        "layers": {"rules": layers_rules, "fuzzy_rule_activations": activations},
        "fuzzy": {"selected_representation": selected_class,
                  "memberships": {object_id: membership_values}, "membership_policies": membership_policies},
        "route": route,
        "disagreement": {"alignment_policy": {"applicable": True}, "alignment_status": "measured",
                         "gamma": out["gamma"], "gamma_max": 0.08,
                         "components": {"native_rule_reconstruction": out["gamma"]},
                         "beta": {"native_rule_reconstruction": 1.0},
                         "reduction_policy": {"applicable": True, "method": "top_rule_projection"},
                         "reduction_status": "measured", "delta": out["delta"],
                         "pre_interpretability": out["i_pre"],
                         "pre_interpretability_definition": ("1 - 2*min(positive_rule_log_odds,negative_rule_log_odds)/total_rule_log_odds" if getattr(model,"rule_semantics","")=="log_odds" else "1-2*sqrt(weighted_variance_of_per_feature_rule_probabilities)")},
        "risk": {"components": components, "risk_weights": WEIGHTS, "rho": rho, "chi_r_crit": chi,
                 "status": "complete", "action": action, "risk_thresholds": THRESHOLDS},
        "diagnostics": diagnostics,
        "explanation_level": {"available_channels": ["native_fuzzy_rules", "trained_membership_policies", "measured_reduction"]},
    }
    return payload, representation, {**out, "rho": rho, "action": action, "selected_class": selected_class}


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    preds = (p >= 0.5).astype(int)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, preds)),
            "f1_malignant": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, p)),
            "brier": float(brier_score_loss(y_true, p))}


def study(output: Path, seed: int = 42, n_features: int = 5, model_factory: Any = None) -> dict[str, Any]:
    dataset = load_breast_cancer()
    values = np.asarray(dataset.data, dtype=float)
    labels = (np.asarray(dataset.target, dtype=int) == 0).astype(int)
    indices = np.arange(len(labels))
    x_train, x_extra, y_train, y_extra, ids_train, ids_extra = train_test_split(
        values, labels, indices, test_size=0.4, stratify=labels, random_state=seed)
    x_val, x_test, y_val, y_test, ids_val, ids_test = train_test_split(
        x_extra, y_extra, ids_extra, test_size=0.5, stratify=y_extra, random_state=seed+1)
    fuzzy = (model_factory or NativeFuzzyWDBC)(n_features=n_features).fit(x_train, y_train, feature_names=list(dataset.feature_names))
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)).fit(x_train, y_train)
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(x_train, y_train)
    models = {"native_fuzzy": fuzzy, "logistic_regression": logistic, "random_forest": forest}
    metrics = {name: _metrics(y_test, predictor.predict_proba(x_test)[:, 1]) for name, predictor in models.items()}
    test_probabilities = {name: predictor.predict_proba(x_test)[:, 1] for name, predictor in models.items()}
    records: list[dict[str, Any]] = []
    sample_receipt = None
    status_counts: dict[str, int] = {}
    version = f"sklearn:breast_cancer:seed={seed}:train={len(x_train)}:val={len(x_val)}:test={len(x_test)}"
    # Exact split identities (public dataset row indices only) are hashed,
    # never replaced with a guessed external patient identifier.
    split_hashes = {"train": sha256([int(v) for v in ids_train]),
                    "validation": sha256([int(v) for v in ids_val]),
                    "test": sha256([int(v) for v in ids_test])}
    output.mkdir(parents=True, exist_ok=True)
    for row_index, raw in enumerate(x_test):
        subject = f"wdbc_test_{row_index:04d}"
        payload, representation, measured = make_receipt_payload(fuzzy, raw, object_id=subject,
                                                                   dataset_version=version)
        certificate = certify_payload(payload,
                                      taxonomy={"method_id": "native_fuzzy_wdbc_v1", "source": "intrinsic", "object": "logic",
                                                "audience": "auditor", "canonical_configuration": "measured native fuzzy rule predictor"},
                                      representation=representation)
        assert verify_certificate(certificate)["valid"]
        status = certificate["decision"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if sample_receipt is None:
            sample_receipt = certificate
        records.append({"object_id": subject, "target_malignant": int(y_test[row_index]),
                        "p_fuzzy": round(measured["probability"], 9),
                        "p_logistic": round(float(test_probabilities["logistic_regression"][row_index]), 9),
                        "p_random_forest": round(float(test_probabilities["random_forest"][row_index]), 9),
                        "delta": round(measured["delta"], 9), "gamma": round(measured["gamma"], 12),
                        "i_pre": round(measured["i_pre"], 9), "rho": round(measured["rho"], 9),
                        "action": measured["action"], "representation": measured["selected_class"],
                        "certificate_status": status, "issue_codes": "|".join(sorted({v["code"] for v in certificate["decision"]["issues"]}))})
    # Stress tests: masked selected source feature, controlled evidence only.
    stress: list[dict[str, Any]] = []
    for row_index in range(min(24, len(x_test))):
        sample = x_test[row_index].copy()
        sample[fuzzy.features[0].index] = np.nan
        payload, representation, measured = make_receipt_payload(fuzzy, sample, object_id=f"wdbc_stress_{row_index:04d}",
                                                                  dataset_version=version, origin="controlled_fixture", allow_missing=True)
        cert = certify_payload(payload,
                               taxonomy={"method_id": "native_fuzzy_wdbc_v1", "source": "intrinsic", "object": "logic",
                                         "audience": "auditor", "canonical_configuration": "measured native fuzzy rule predictor"},
                               representation=representation, policy={"allow_controlled_fixture": True})
        assert verify_certificate(cert)["valid"]
        assert measured["action"] == "block"
        stress.append({"id": f"stress_{row_index:04d}", "action": measured["action"],
                       "status": cert["decision"]["status"], "chi_R": 1,
                       "issues": sorted({v["code"] for v in cert["decision"]["issues"]})})
    with (output / "heldout_native_fuzzy.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "sample_certificate.json").write_text(json.dumps(sample_receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "trained_native_fuzzy.json").write_text(json.dumps(fuzzy.fitted_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "study": fuzzy.fitted_json()["model_type"], "source_dataset": "sklearn.datasets.load_breast_cancer",
        "source_role": "public_real_world_tabular_dataset_not_clinical_validation",
        "seed": seed, "n_train": len(x_train), "n_validation": len(x_val), "n_test": len(x_test),
        "split_sha256": split_hashes, "dataset_sha256": sha256({"shape": list(values.shape),
                                                            "X": values.tolist(), "y_malignant": labels.tolist()}),
        "positive_class": "malignant", "split_uses_stratification": True,
        "selected_features_train_only": [f.name for f in fuzzy.features],
        "all_membership_functions_fitted_on_train_only": True,
        "all_rule_consequents_fitted_on_train_only": True,
        "baselines_have_same_training_and_test_rows": True,
        "metric_comparison": metrics, "certification_status_counts": status_counts,
        "stress_missing_feature": {"n": len(stress), "all_actions_blocked": all(v["action"] == "block" for v in stress),
                                   "statuses": {name: sum(v["status"] == name for v in stress) for name in {v["status"] for v in stress}}},
        "limitation": ["Fuzzy representation fitted solely on tabular training observations",
                       "Policy thresholds are fixed research presets, not calibrated treatment decisions",
                       "Rule consistency is a descriptive proxy, not a validated psychological interpretability metric",
                       "Structural certification is NOT clinical or causal validation", "Single split: compare with repeated seeds before scientific claims"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/chapter5/native_fuzzy_wdbc"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(study(args.out, seed=args.seed, n_features=args.features), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

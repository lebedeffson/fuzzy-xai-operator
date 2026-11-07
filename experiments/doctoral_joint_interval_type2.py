"""Exploratory jointly fitted interval type-2 fuzzy classifier and honest baselines.

The existing IT2 model fits an ordinary fuzzy-logistic classifier and then
bootstraps a footprint of uncertainty around its fixed membership functions.
Here the rule contrasts, nominal membership knots and FOU widths all enter ONE
predeclared supervised objective and are optimized together. This is a local
L-BFGS-B optimization with finite-difference gradients, NOT a proof of global
optimality. It is a training algorithm, not post-hoc adjustment of a receipt.

FOU bounds quantify parameter variation, never predictive/clinical coverage.
The WDBC splits overlap across seeds and are descriptive, not independent trials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.interval_type2 import (
    IT2FeatureRule, IT2FuzzyLogit, IT2Shoulder, content_sha256,
    make_it2_receipt, verify_it2_receipt,
)
from fuzzyxai.it2_conformal import IT2SplitConformal
from .doctoral_interval_type2 import NativeIntervalType2Logistic

SCHEMA = "fuzzyxai-joint-it2-exploratory-v1"


def _checksums(root: Path) -> None:
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest_sha256.json"
    }
    (root / "manifest_sha256.json").write_text(
        json.dumps({"schema": "sha256-manifest-v1", "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )


class JointIntervalType2Logistic:
    """Jointly optimize FOU knots/widths and fuzzy-logit rule coefficients.

    The selected features and the bootstrap initial FOU are fitted on training
    only. Radius lower bounds make this a restricted, non-degenerate IT2 family;
    no claim that these lower bounds are statistically calibrated is made.
    """

    def __init__(self, *, n_features: int = 5, n_bootstrap: int = 16,
                 seed: int = 42, max_iter: int = 300,
                 robust_weight: float = 0.18, coefficient_l2: float = 0.002,
                 width_prior_weight: float = 4.0, shift_penalty: float = 0.04):
        if isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if n_bootstrap < 4 or any(not np.isfinite(v) or v < 0 for v in
             (robust_weight, coefficient_l2, width_prior_weight, shift_penalty)):
            raise ValueError("invalid fixed objective coefficients or bootstraps")
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.max_iter = max_iter
        self.robust_weight = float(robust_weight)
        self.coefficient_l2 = float(coefficient_l2)
        self.width_prior_weight = float(width_prior_weight)
        self.shift_penalty = float(shift_penalty)
        self.baseline: NativeIntervalType2Logistic | None = None
        self.it2: IT2FuzzyLogit | None = None
        self.training_receipt: dict[str, Any] | None = None

    @staticmethod
    def _vectorized_logits(
        vector: np.ndarray, X_selected: np.ndarray, left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exactly reproduce the paired-rule, coupled-antecedent IT2 semantics."""
        f = len(left)
        intercept = vector[0]
        contrasts = vector[1:1+f]
        shift_left = vector[1+f:1+2*f]
        shift_right = vector[1+2*f:1+3*f]
        radius_left = vector[1+3*f:1+4*f]
        radius_right = vector[1+4*f:1+5*f]
        gap = right - left
        l_nom = left + gap * shift_left
        r_nom = right + gap * shift_right
        l_min = l_nom - gap * radius_left
        l_max = l_nom + gap * radius_left
        r_min = r_nom - gap * radius_right
        r_max = r_nom + gap * radius_right
        if np.any(l_max >= r_min):
            raise ValueError("learned membership parameter ranges overlap")
        # MU_high is non-increasing in both knot coordinates. Intervals of
        # paired low/high memberships are correlated, not independent.
        m_lower = np.clip((X_selected-l_max)/(r_max-l_max), 0, 1)
        m_upper = np.clip((X_selected-l_min)/(r_min-l_min), 0, 1)
        m_nom = np.clip((X_selected-l_nom)/(r_nom-l_nom), 0, 1)
        contributions_lo = np.minimum(contrasts*m_lower, contrasts*m_upper)
        contributions_hi = np.maximum(contrasts*m_lower, contrasts*m_upper)
        return (
            intercept + np.sum(contributions_lo, axis=1),
            intercept + np.sum(contributions_hi, axis=1),
            intercept + np.sum(contrasts*m_nom, axis=1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray,
            feature_names: list[str]) -> "JointIntervalType2Logistic":
        values = np.asarray(x, dtype=float)
        labels = np.asarray(y, dtype=int)
        if values.ndim != 2 or labels.shape != (len(values),) or len(values) < 30:
            raise ValueError("input must be an adequately sized binary training matrix")
        if not np.isfinite(values).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError("fit requires finite values and both class labels")
        baseline = NativeIntervalType2Logistic(
            n_features=self.n_features, n_bootstrap=self.n_bootstrap, seed=self.seed,
        ).fit(values, labels, feature_names)
        if baseline.it2 is None:
            raise AssertionError("baseline IT2 FOU was not fitted")
        f = len(baseline.it2.features)
        indices = np.asarray([feature.feature_index for feature in baseline.it2.features])
        original_left = np.asarray([feature.membership.nominal_left for feature in baseline.it2.features])
        original_right = np.asarray([feature.membership.nominal_right for feature in baseline.it2.features])
        span = original_right - original_left
        if np.any(span <= 0):
            raise AssertionError("training source contains a degenerate membership interval")
        contrasts = np.asarray([
            feature.high_coefficient - feature.low_coefficient
            for feature in baseline.it2.features
        ])
        initial_intercept = float(baseline.it2.intercept + sum(
            feature.low_coefficient for feature in baseline.it2.features
        ))
        # Use empirical bootstrap FOU as a train-only prior, but allow the
        # uncertainty widths to move under the *same* supervised objective.
        radius_left = np.asarray([
            max(feature.membership.nominal_left-feature.membership.left_min,
                feature.membership.left_max-feature.membership.nominal_left)/gap
            for feature, gap in zip(baseline.it2.features, span)
        ])
        radius_right = np.asarray([
            max(feature.membership.nominal_right-feature.membership.right_min,
                feature.membership.right_max-feature.membership.nominal_right)/gap
            for feature, gap in zip(baseline.it2.features, span)
        ])
        prior_left = np.clip(radius_left, .012, .16)
        prior_right = np.clip(radius_right, .012, .16)
        x0 = np.concatenate([
            [initial_intercept], contrasts, np.zeros(2*f), prior_left, prior_right,
        ])
        xsel = values[:, indices]
        bounds = [(initial_intercept-12, initial_intercept+12)] + [(-40., 40.)]*f \
               + [(-.12, .12)]*(2*f) + [(.012, .16)]*(2*f)

        def objective(vector: np.ndarray) -> float:
            zlo, zhi, _ = self._vectorized_logits(vector, xsel, original_left, original_right)
            midpoint = (expit(zlo) + expit(zhi))/2.
            clipped = np.clip(midpoint, 1e-12, 1-1e-12)
            logloss = -np.mean(labels*np.log(clipped)+(1-labels)*np.log1p(-clipped))
            # Worst-case BCE with IT2 logit interval: y=1 takes the lower
            # bound; y=0 the upper bound. This is NOT a conformal objective.
            worst = np.mean(labels*np.logaddexp(0., -zlo)
                            + (1-labels)*np.logaddexp(0., zhi))
            diff_width = np.r_[vector[1+3*f:1+4*f]-prior_left,
                               vector[1+4*f:1+5*f]-prior_right]
            return float(logloss + self.robust_weight*worst
                         + self.coefficient_l2*np.sum(vector[1:1+f]**2)
                         + self.shift_penalty*np.sum(vector[1+f:1+3*f]**2)
                         + self.width_prior_weight*np.sum(diff_width**2))

        initial_loss = objective(x0)
        solution = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": self.max_iter, "ftol": 1e-9})
        if not np.isfinite(solution.fun) or solution.fun > initial_loss + 1e-8:
            raise ArithmeticError("joint optimization failed to improve its training objective")
        vector = np.asarray(solution.x, dtype=float)
        gap = original_right - original_left
        l_nom = original_left + gap*vector[1+f:1+2*f]
        r_nom = original_right + gap*vector[1+2*f:1+3*f]
        rl = gap*vector[1+3*f:1+4*f]
        rr = gap*vector[1+4*f:1+5*f]
        features = tuple(
            IT2FeatureRule(
                feature_index=int(index), feature_name=feature.feature_name,
                low_coefficient=0., high_coefficient=float(vector[1+i]),
                membership=IT2Shoulder(
                    left_min=float(l_nom[i]-rl[i]), left_max=float(l_nom[i]+rl[i]),
                    right_min=float(r_nom[i]-rr[i]), right_max=float(r_nom[i]+rr[i]),
                    nominal_left=float(l_nom[i]), nominal_right=float(r_nom[i]),
                ),
            )
            for i, (index, feature) in enumerate(zip(indices, baseline.it2.features))
        )
        self.it2 = IT2FuzzyLogit(
            intercept=float(vector[0]), n_input_features=values.shape[1],
            features=features,
            source_fit="train_only_joint_robust_BCE_membership_knots_widths_rule_contrasts",
        )
        self.baseline = baseline
        self.training_receipt = {
            "model_sha256": content_sha256(self.it2.to_dict()),
            "baseline_sha256": content_sha256(baseline.it2.to_dict()),
            "training_data_sha256": content_sha256({"X": values.tolist(), "y": labels.tolist()}),
            "optimizer": "scipy_LBFGSB_joint_finite_difference",
            "optimizer_success": bool(solution.success),
            "optimizer_message": str(solution.message),
            "optimizer_nit": int(solution.nit),
            "initial_objective": initial_loss,
            "final_objective": float(solution.fun),
            "max_iter": self.max_iter,
            "n_features": f,
            "n_training": len(values),
            "robust_weight": self.robust_weight,
            "width_prior_weight": self.width_prior_weight,
            "shift_penalty": self.shift_penalty,
            "coefficient_l2": self.coefficient_l2,
            "radii_lower_bound": .012,
            "radii_upper_bound": .16,
            "parameter_family": "saturated_paired_IT2_shoulders_with_coupled_low_high",
            "max_knot_center_shift_in_training_span": .12,
            "all_parameters_in_one_objective": True,
            "limitations": [
                "Nonconvex optimization; convergence is local and finite-difference-based",
                "Feature selection and bootstrap initialization from the same train split",
                "FOU widths have imposed nonzero bounds, not a learned coverage guarantee",
                "Bootstrap prior and numerical coefficient regularization are fixed design choices",
            ],
        }
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.it2 is None:
            raise RuntimeError("fit the model first")
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.it2.n_input_features or not np.isfinite(values).all():
            raise ValueError("finite input matching the trained feature schema required")
        # Reference inference rather than a second, unverified predictor path.
        p = np.asarray([self.it2.infer(row)["probability_midpoint"] for row in values])
        return np.column_stack((1-p, p))


def _ids(indices: np.ndarray) -> list[str]:
    return [f"wdbc-sklearn-row-{int(i)}" for i in indices]


def _decision_metrics(labels: np.ndarray, probs: np.ndarray,
                      accepts: np.ndarray, sets: list[dict[str, Any]]) -> dict[str, float | int | None]:
    pred = (probs >= .5).astype(int)
    nacc = int(np.sum(accepts))
    nerr = int(np.sum((pred != labels)&accepts))
    sizes = np.asarray([len(s["label_set"]) for s in sets], dtype=int)
    return {
        "n_test": int(len(labels)), "auto_count": nacc,
        "auto_fraction": nacc/len(labels), "auto_errors": nerr,
        "selective_error": nerr/nacc if nacc else None,
        "false_negatives_accepted": int(np.sum((labels==1)&(pred==0)&accepts)),
        "empirical_set_coverage": float(np.mean([int(labels[i]) in sets[i]["label_set"] for i in range(len(labels))])),
        "mean_set_size": float(np.mean(sizes)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
    }


def _singleton(sets: list[dict[str, Any]], probs: np.ndarray) -> np.ndarray:
    return np.asarray([len(s["label_set"]) == 1 and s["label_set"][0] == int(p >= .5)
                       for s, p in zip(sets, probs)], dtype=bool)


def _matched_margin(policy_t1: np.ndarray, policy_t1_sets: list[dict[str, Any]],
                    target_acceptance: float) -> float:
    eligible = _singleton(policy_t1_sets, policy_t1)
    margins = np.abs(policy_t1-.5)
    options = [-1.] + sorted(set(map(float, margins)))
    return float(min(options, key=lambda t:
        (abs(float(np.mean(eligible & (margins>t))) - target_acceptance), -t)))


def run_one(seed: int, root: Path, *, n_bootstrap: int = 16, max_iter: int = 300,
            alpha: float = .1, stress_levels: tuple[float, ...] = (0., .10, .25)) -> dict[str, Any]:
    dataset = load_breast_cancer()
    values = np.asarray(dataset.data, dtype=float)
    labels = (np.asarray(dataset.target) == 0).astype(int)
    id_all = np.arange(len(labels))
    xtr, xother, ytr, yother, itr, iother = train_test_split(
        values, labels, id_all, test_size=.4, stratify=labels, random_state=seed)
    xval, xte, yval, yte, ival, ite = train_test_split(
        xother, yother, iother, test_size=.5, stratify=yother, random_state=seed+1)
    xcal, xpolicy, ycal, ypolicy, ical, ipolicy = train_test_split(
        xval, yval, ival, test_size=.3, stratify=yval, random_state=seed+2)
    split = [set(_ids(s)) for s in (itr, ical, ipolicy, ite)]
    if not all(split[i].isdisjoint(split[j]) for i in range(4) for j in range(i+1,4)):
        raise AssertionError("train/calibration/policy/test source identities overlap")
    if sum(map(len,split)) != len(labels):
        raise AssertionError("source split count differs from dataset")
    learned = JointIntervalType2Logistic(
        seed=seed, n_bootstrap=n_bootstrap, max_iter=max_iter).fit(xtr, ytr, list(dataset.feature_names))
    if learned.it2 is None or learned.baseline is None or learned.baseline.it2 is None:
        raise AssertionError("IT2 fitting incomplete")
    baseline = learned.baseline
    models = {"joint_it2": learned, "bootstrap_it2": baseline}
    training_ids, cal_ids, policy_ids, test_ids = [_ids(z) for z in (itr, ical, ipolicy, ite)]
    calibrations = {
        name: IT2SplitConformal.fit_it2(
            model.it2, xcal.tolist(), ycal.tolist(), training_ids=training_ids,
            calibration_ids=cal_ids,
            calibration_dataset_version=f"wdbc:seed={seed}:calibration",
            alpha=alpha,
        ) for name, model in models.items()
    }
    t1_cal = IT2SplitConformal.fit_probabilities(
        baseline.type1.predict_proba(xcal)[:,1].tolist(), ycal.tolist(),
        model_sha256=content_sha256(baseline.type1.fitted_json()),
        calibration_ids=cal_ids, training_ids=training_ids,
        calibration_dataset_version=f"wdbc:seed={seed}:calibration", alpha=alpha,
    )
    policy_jp = learned.predict_proba(xpolicy)[:,1]
    policy_js = [calibrations["joint_it2"].predict_set(float(p),object_id=oid)
                 for p,oid in zip(policy_jp,policy_ids)]
    policy_safe = np.asarray([
        learned.it2.infer(row)["probability_lower"] >= .5
        or learned.it2.infer(row)["probability_upper"] < .5
        for row in xpolicy],dtype=bool)
    policy_width = np.asarray([learned.it2.infer(row)["induced_width"] <= .15 for row in xpolicy])
    policy_joint_accept = _singleton(policy_js,policy_jp)&policy_safe&policy_width
    policy_t1 = baseline.type1.predict_proba(xpolicy)[:,1]
    policy_sets_t1 = [t1_cal.predict_set(float(p),object_id=oid)
                      for p,oid in zip(policy_t1,policy_ids)]
    margin = _matched_margin(policy_t1, policy_sets_t1, float(policy_joint_accept.mean()))
    train_std = np.maximum(np.std(xtr,axis=0),1e-9)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    results: dict[str,Any] = {}
    for noise in stress_levels:
        if not (0 <= noise <= 1 and np.isfinite(noise)):
            raise ValueError("synthetic corruption scale must lie in [0,1]")
        rng=np.random.default_rng(seed*1001+int(noise*10000))
        # Measurement corruption only: reuse labels, never claim a real
        # independent population or a conformal guarantee under shift.
        test_x = xte if noise == 0 else xte+rng.normal(0.,noise*train_std,size=xte.shape)
        probabilities = {
            "type1_matched": baseline.type1.predict_proba(test_x)[:,1],
            "bootstrap_it2": baseline.predict_proba(test_x)[:,1],
            "joint_it2": learned.predict_proba(test_x)[:,1],
        }
        run_results={}
        for name, probs in probabilities.items():
            conformal = t1_cal if name == "type1_matched" else calibrations[name]
            sets=[conformal.predict_set(float(p),object_id=oid)
                  for p,oid in zip(probs,test_ids)]
            singleton=_singleton(sets,probs)
            evidence = None
            if name == "type1_matched":
                accepted=singleton & (np.abs(probs-.5)>margin)
            else:
                model=models[name].it2
                evidence=[model.infer(row) for row in test_x]
                guard=np.asarray([e["probability_lower"]>=.5 or e["probability_upper"]<.5 for e in evidence])
                width=np.asarray([e["induced_width"]<=.15 for e in evidence])
                accepted=singleton&guard&width
            run_results[name]=_decision_metrics(yte,probs,accepted,sets)
            for i in range(len(yte)):
                records.append({
                    "seed": seed, "row_id": test_ids[i], "synthetic_noise_std_fraction": noise,
                    "method": name, "true_malignant": int(yte[i]), "p_malignant": float(probs[i]),
                    "prediction_malignant": int(probs[i]>=.5), "auto_accept": int(accepted[i]),
                    "error_if_auto": int(accepted[i] and ((probs[i]>=.5)!=bool(yte[i]))),
                    "conformal_set": "|".join(map(str,sets[i]["label_set"])),
                    "fou_probability_lower": float(evidence[i]["probability_lower"]) if evidence is not None else None,
                    "fou_probability_upper": float(evidence[i]["probability_upper"]) if evidence is not None else None,
                    "fou_width": float(evidence[i]["induced_width"]) if evidence is not None else None,
                    "fou_crosses_threshold": int(
                        evidence[i]["probability_lower"] < .5 <= evidence[i]["probability_upper"]
                    ) if evidence is not None else None,
                    "blocked_reason": (
                        "accepted" if accepted[i] else
                        "conformal_ambiguous" if not singleton[i] else
                        "type1_matched_margin" if evidence is None else
                        "fou_crosses_threshold" if evidence[i]["probability_lower"] < .5 <= evidence[i]["probability_upper"] else
                        "fou_excessive_width" if evidence[i]["induced_width"] > .15 else
                        "other_policy_block"
                    ),
                })
        results[str(noise)]=run_results
    with (root/"heldout_and_corruption.csv").open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (root/"joint_model.json").write_text(json.dumps(learned.it2.to_dict(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"joint_fit_receipt.json").write_text(json.dumps(learned.training_receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"calibration_manifest.json").write_text(json.dumps({k:v.to_dict() for k,v in {**calibrations,"type1_matched":t1_cal}.items()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    receipt=make_it2_receipt(learned.it2,xte[0].tolist(),source_id=test_ids[0],dataset_version=f"wdbc:seed={seed}:test")
    if not verify_it2_receipt(receipt,trusted_model_sha256=content_sha256(learned.it2.to_dict()))["valid"]:
        raise AssertionError("joint Type-2 model numeric certificate could not be independently replayed")
    (root/"sample_model_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={
        "schema":SCHEMA,"seed":seed,"n_train":len(itr),"n_calibration":len(ical),
        "n_policy":len(ipolicy),"n_test":len(ite),"split_sha256":{
            label:content_sha256(sorted(ids)) for label,ids in zip(("train","calibration","policy","test"),split)},
        "model_receipt_sha256":content_sha256(learned.it2.to_dict()),
        "objective_training_only":learned.training_receipt,
        "policy_target_accept_fraction":float(policy_joint_accept.mean()),
        "t1_matched_margin_on_policy":margin,
        "results":results,
        "limitations":[
            "Repeated WDBC seeds use overlapping test objects, not independent clinical trials",
            "Bootstrap baseline IT2 and jointly fitted IT2 differ in parameters, not in trained feature-selection data",
            "Synthetic Gaussian measurement corruption is NOT an independent dataset or true distribution shift",
            "Matched coverage selected on separate policy data; realized heldout rates may differ",
            "Conformal exchangeability not proven under class-stratified split or corruption",
            "FOU widths are NOT calibrated predictive confidence intervals",
        ],
    }
    (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary


def run(root:Path,*,seeds:tuple[int,...]=(17,42,73,123,2026),n_bootstrap:int=16,max_iter:int=300,
        stress_levels:tuple[float,...]=(0.,.10,.25))->dict[str,Any]:
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("seeds must be unique and nonempty")
    root.mkdir(parents=True,exist_ok=True)
    splits=[run_one(seed,root/f"seed_{seed}",n_bootstrap=n_bootstrap,
                    max_iter=max_iter,stress_levels=stress_levels) for seed in seeds]
    aggregate={}
    for level in stress_levels:
        stats={}
        for method in ("joint_it2","bootstrap_it2","type1_matched"):
            ms=[row["results"][str(level)][method] for row in splits]
            accepted=sum(m["auto_count"] for m in ms)
            errors=sum(m["auto_errors"] for m in ms)
            stats[method]={
                "automatic_accepted_repeated_rows":accepted,
                "errors_on_accepted_repeated_rows":errors,
                "selective_error_pooled_descriptive":errors/accepted if accepted else None,
                "automatic_accept_mean":mean(m["auto_fraction"] for m in ms),
                "roc_auc_mean":mean(m["roc_auc"] for m in ms),
                "brier_mean":mean(m["brier"] for m in ms),
                "mean_conformal_set_coverage":mean(m["empirical_set_coverage"] for m in ms),
            }
        aggregate[str(level)]=stats
    summary={
        "schema":SCHEMA,"seed_list":list(seeds),"n_bootstrap":n_bootstrap,
        "max_iter":max_iter,"stress_levels":list(stress_levels),
        "aggregates_descriptive":aggregate,
        "per_split":splits,
        "interpretation":"All observed metrics are exploratory and no model superiority is presumed",
        "independence_warning":"Five overlapping WDBC test splits are not independent clinical evaluations",
    }
    (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    _checksums(root)
    return summary


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("reports/chapter5/joint_type2"))
    p.add_argument("--seeds",nargs="+",type=int,default=[17,42,73,123,2026])
    p.add_argument("--bootstrap",type=int,default=16)
    p.add_argument("--max-iter",type=int,default=300)
    a=p.parse_args()
    print(json.dumps(run(a.out,seeds=tuple(a.seeds),n_bootstrap=a.bootstrap,max_iter=a.max_iter),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

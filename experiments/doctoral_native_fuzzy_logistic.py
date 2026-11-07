"""Fuzzy-concept logistic predictor: rules participate in model inference.

The feature memberships are fit on training inputs only. A supervised logistic
layer is fit on actual [low, high] fuzzy degrees, not on original raw features.
Every prediction is reconstructed from those rule activations and learned
coefficients; the explanation is not a post-hoc approximation.

This is an exploratory transparent-ML benchmark, not clinical certification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from .doctoral_native_fuzzy_wdbc import FuzzyFeature, NativeFuzzyWDBC, study


class NativeFuzzyLogistic(NativeFuzzyWDBC):
    """Logit(x) = intercept + sum_r coefficient_r * membership_r(x)."""

    def __init__(self, n_features: int = 5, regularization_c: float = 1.0,
                 reduction_keep_fraction: float = 0.75):
        super().__init__(n_features=n_features)
        if not 0 < regularization_c or not math.isfinite(regularization_c):
            raise ValueError("regularization C must be positive and finite")
        if not 0 < reduction_keep_fraction <= 1:
            raise ValueError("reduction fraction must be in (0,1]")
        self.regularization_c = float(regularization_c)
        self.reduction_keep_fraction = float(reduction_keep_fraction)
        self.rule_semantics = "log_odds"
        self.clf: LogisticRegression | None = None
        self.intercept = 0.0

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if not self.features:
            raise RuntimeError("fit model first")
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or not np.isfinite(values).all():
            raise ValueError("input features must be a finite training-schema matrix")
        columns = []
        for feature in self.features:
            high = np.clip((values[:, feature.index] - feature.low)/(feature.high-feature.low), 0., 1.)
            columns.extend([1. - high, high])
        return np.column_stack(columns)

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> NativeFuzzyLogistic:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 30 or not np.isfinite(x).all():
            raise ValueError("finite training matrix and matching labels required")
        if not set(np.unique(y)).issubset({0,1}) or len(np.unique(y)) != 2:
            raise ValueError("two binary classes required")
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(x.shape[1])])
        if len(self.feature_names) != x.shape[1]:
            raise ValueError("feature names mismatch")
        self.medians = [float(v) for v in np.median(x, axis=0)]
        self.train_size, self.positive_prior = len(x), float(y.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            f_scores, _ = f_classif(x, y)
        scores = np.where(np.isfinite(f_scores), np.maximum(0., f_scores), 0.)
        candidates = np.argsort(-scores, kind="stable")
        chosen: list[FuzzyFeature] = []
        for j in candidates:
            lo, hi = np.quantile(x[:, j], [.02,.98])
            if hi <= lo + 1e-10:
                continue
            high = np.clip((x[:, j] - lo) / (hi - lo), 0.,1.)
            chosen.append(FuzzyFeature(int(j), self.feature_names[j], float(lo), float(hi),
                                       1.,0.,0.,float((1.-high).sum()),float(high.sum())))
            if len(chosen) == self.n_features:
                break
        if not chosen:
            raise ValueError("no variable source features")
        self.features = chosen
        self.clf = LogisticRegression(C=self.regularization_c, max_iter=2000, random_state=0).fit(self._transform(x), y)
        self.intercept = float(self.clf.intercept_[0])
        self.features = [FuzzyFeature(item.index, item.name, item.low, item.high, 1.,
                                      float(self.clf.coef_[0, 2*i]), float(self.clf.coef_[0, 2*i+1]),
                                      item.low_support, item.high_support)
                         for i, item in enumerate(self.features)]
        return self

    def fitted_json(self) -> dict[str, Any]:
        result = super().fitted_json()
        result.update({"model_type": "NativeFuzzyLogistic-v2", "consequent_semantics": "signed_log_odds",
                       "regularization_c": self.regularization_c, "intercept_log_odds": self.intercept,
                       "reduction_keep_fraction": self.reduction_keep_fraction,
                       "model_fit": "supervised_logistic_on_train_fuzzy_degrees"})
        return result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit model first")
        return self.clf.predict_proba(self._transform(x))

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0+math.exp(-value)) if value >= 0 else math.exp(value)/(1.0+math.exp(value))

    def _explain(self, row: np.ndarray, *, allow_missing: bool = False) -> dict[str, Any]:
        if self.clf is None:
            raise RuntimeError("fit model first")
        raw = np.asarray(row, dtype=float)
        if raw.shape != (len(self.feature_names),) or np.isinf(raw).any():
            raise ValueError("invalid source feature vector")
        missing = np.flatnonzero(np.isnan(raw)).tolist()
        if missing and not allow_missing:
            raise ValueError("missing feature requires explicit critical override")
        used = raw.copy()
        if missing:
            used[missing] = np.asarray(self.medians)[missing]
        rules: list[dict[str, Any]] = []
        for feat in self.features:
            raw_value = float(used[feat.index])
            value = min(max(raw_value, feat.low), feat.high)
            lo, hi = feat.membership(value)
            for term, degree, coeff in (("low",lo,feat.low_consequent),("high",hi,feat.high_consequent)):
                contribution = float(degree*coeff)
                rules.append({"rule_id":f"{feat.key}:{term}","feature":feat.key,"feature_name":feat.name,
                              "term":term,"value":value,"raw_value":raw_value,
                              "preprocessing":"clip_to_train_fitted_interval",
                              "membership":float(degree),"feature_weight":1.,
                              "consequent":float(coeff),"contribution":contribution,"strength":abs(contribution)})
        logit = self.intercept + sum(v["contribution"] for v in rules)
        reconstructed = self._sigmoid(logit)
        actual = float(self.predict_proba(used)[0,1])
        gamma = abs(reconstructed - actual)
        supporting = sum(max(0.,v["contribution"]) for v in rules)
        opposing = sum(max(0.,-v["contribution"]) for v in rules)
        i_pre = 1. - (2.*min(supporting,opposing)/(supporting+opposing)) if supporting+opposing>1e-15 else 1.
        entropy = 0. if actual in (0.,1.) else -(actual*math.log2(actual)+(1.-actual)*math.log2(1.-actual))
        keep = max(1, math.ceil(len(rules)*self.reduction_keep_fraction))
        top = sorted(rules, key=lambda v: (v["strength"],v["rule_id"]),reverse=True)[:keep]
        reduced = self._sigmoid(self.intercept+sum(v["contribution"] for v in top))
        return {"probability":actual,"raw":raw,"used":used,"missing":missing,"rules":rules,"top_rules":top,
                "gamma":gamma,"delta":abs(actual-reduced),"score_reduced":reduced,"i_pre":i_pre,
                "uncertainty":entropy,"full_logit":logit,"intercept":self.intercept}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path('reports/chapter5/native_fuzzy_logistic'))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--features",type=int,default=5)
    args=p.parse_args()
    print(json.dumps(study(args.out,seed=args.seed,n_features=args.features,model_factory=NativeFuzzyLogistic),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

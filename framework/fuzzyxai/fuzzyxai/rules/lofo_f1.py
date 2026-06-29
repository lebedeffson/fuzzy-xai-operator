from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class RuleImportance:
    """Single-rule importance measured by F1 drop after removing the rule."""

    rule_index: int
    rule_name: str
    delta_f1: float
    f1_without: float
    baseline_f1: float
    coefficient: float
    mean_abs_activation: float


@dataclass(frozen=True)
class BootstrapRuleImportance:
    """Bootstrap-aggregated LOFO-F1 importance for stable top-B selection."""

    rule_index: int
    rule_name: str
    mean_delta_f1: float
    std_delta_f1: float
    top_frequency: float
    mean_rank: float


def _as_2d_matrix(h: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(h, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('H_val must be a 2D matrix with shape (n_samples, n_rules)')
    return matrix


def _rule_names(n_rules: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [f'rule_{idx}' for idx in range(n_rules)]
    if len(names) != n_rules:
        raise ValueError('rule_names length must match number of rules')
    return [str(name) for name in names]


def logits_from_rules(h_val: np.ndarray | Sequence[Sequence[float]], theta: np.ndarray | Sequence[float], bias: float = 0.0) -> np.ndarray:
    """Compute additive KAFN-style logits: bias + H @ theta."""
    h = _as_2d_matrix(h_val)
    coef = np.asarray(theta, dtype=float)
    if coef.ndim != 1 or coef.shape[0] != h.shape[1]:
        raise ValueError('theta must be a 1D vector with length n_rules')
    return float(bias) + h @ coef


def binary_predictions_from_logits(logits: np.ndarray | Sequence[float], threshold: float = 0.0) -> np.ndarray:
    """Convert logits to binary labels; threshold 0 is equivalent to probability 0.5."""
    return (np.asarray(logits, dtype=float) >= float(threshold)).astype(int)


def lofo_f1_importance(
    h_val: np.ndarray | Sequence[Sequence[float]],
    theta: np.ndarray | Sequence[float],
    y_val: np.ndarray | Sequence[int],
    *,
    bias: float = 0.0,
    threshold: float = 0.0,
    rule_names: Sequence[str] | None = None,
    sample_weight: np.ndarray | Sequence[float] | None = None,
) -> list[RuleImportance]:
    """Rank rules by Leave-One-Rule-Out F1 drop without retraining.

    For each rule r the function computes:
        z_without_r = z_full - H_val[:, r] * theta[r]
        delta_f1 = F1(z_full) - F1(z_without_r)

    Positive delta means the rule helps F1. Negative delta means the rule hurts F1.
    """
    h = _as_2d_matrix(h_val)
    coef = np.asarray(theta, dtype=float)
    y = np.asarray(y_val, dtype=int)
    if y.ndim != 1 or y.shape[0] != h.shape[0]:
        raise ValueError('y_val must be a 1D vector with length n_samples')
    if coef.ndim != 1 or coef.shape[0] != h.shape[1]:
        raise ValueError('theta must be a 1D vector with length n_rules')
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    names = _rule_names(h.shape[1], rule_names)

    z_full = logits_from_rules(h, coef, bias=bias)
    pred_full = binary_predictions_from_logits(z_full, threshold=threshold)
    baseline = float(f1_score(y, pred_full, sample_weight=weights, zero_division=0))

    rows: list[RuleImportance] = []
    for rule_idx in range(h.shape[1]):
        z_without = z_full - h[:, rule_idx] * coef[rule_idx]
        pred_without = binary_predictions_from_logits(z_without, threshold=threshold)
        f1_without = float(f1_score(y, pred_without, sample_weight=weights, zero_division=0))
        rows.append(
            RuleImportance(
                rule_index=rule_idx,
                rule_name=names[rule_idx],
                delta_f1=float(baseline - f1_without),
                f1_without=f1_without,
                baseline_f1=baseline,
                coefficient=float(coef[rule_idx]),
                mean_abs_activation=float(np.mean(np.abs(h[:, rule_idx]))),
            )
        )
    return sorted(rows, key=lambda row: (row.delta_f1, row.mean_abs_activation), reverse=True)


def select_top_rules_by_lofo_f1(importances: Sequence[RuleImportance], budget: int) -> list[int]:
    """Return indices of top-B helpful rules according to LOFO-F1."""
    if budget < 0:
        raise ValueError('budget must be non-negative')
    return [row.rule_index for row in importances[:budget]]


def budget_prune_importance(h_val: np.ndarray | Sequence[Sequence[float]], theta: np.ndarray | Sequence[float]) -> np.ndarray:
    """Old cheap heuristic: |theta_r| * mean(|H[:, r]|)."""
    h = _as_2d_matrix(h_val)
    coef = np.asarray(theta, dtype=float)
    if coef.ndim != 1 or coef.shape[0] != h.shape[1]:
        raise ValueError('theta must be a 1D vector with length n_rules')
    return np.abs(coef) * np.mean(np.abs(h), axis=0)


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    """Jaccard similarity for two selected rule sets."""
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def bootstrap_lofo_f1_importance(
    h_val: np.ndarray | Sequence[Sequence[float]],
    theta: np.ndarray | Sequence[float],
    y_val: np.ndarray | Sequence[int],
    *,
    bias: float = 0.0,
    threshold: float = 0.0,
    rule_names: Sequence[str] | None = None,
    n_bootstraps: int = 50,
    top_k: int | None = None,
    random_state: int = 42,
) -> list[BootstrapRuleImportance]:
    """Bootstrap LOFO-F1 for stable rule ranking.

    `top_frequency` is the share of bootstrap samples where the rule appears in top_k.
    If top_k is omitted, it defaults to 10% of rules, at least one rule.
    """
    h = _as_2d_matrix(h_val)
    y = np.asarray(y_val, dtype=int)
    coef = np.asarray(theta, dtype=float)
    if n_bootstraps <= 0:
        raise ValueError('n_bootstraps must be positive')
    n_rules = h.shape[1]
    top = max(1, int(round(0.10 * n_rules))) if top_k is None else int(top_k)
    top = max(1, min(top, n_rules))
    names = _rule_names(n_rules, rule_names)
    rng = np.random.default_rng(random_state)

    deltas = np.zeros((n_bootstraps, n_rules), dtype=float)
    ranks = np.zeros((n_bootstraps, n_rules), dtype=float)
    top_hits = np.zeros(n_rules, dtype=float)

    for boot_idx in range(n_bootstraps):
        sample_idx = rng.integers(0, h.shape[0], size=h.shape[0])
        rows = lofo_f1_importance(
            h[sample_idx],
            coef,
            y[sample_idx],
            bias=bias,
            threshold=threshold,
            rule_names=names,
        )
        for rank, row in enumerate(rows, start=1):
            deltas[boot_idx, row.rule_index] = row.delta_f1
            ranks[boot_idx, row.rule_index] = rank
            if rank <= top:
                top_hits[row.rule_index] += 1.0

    result = [
        BootstrapRuleImportance(
            rule_index=idx,
            rule_name=names[idx],
            mean_delta_f1=float(np.mean(deltas[:, idx])),
            std_delta_f1=float(np.std(deltas[:, idx], ddof=0)),
            top_frequency=float(top_hits[idx] / n_bootstraps),
            mean_rank=float(np.mean(ranks[:, idx])),
        )
        for idx in range(n_rules)
    ]
    return sorted(result, key=lambda row: (row.top_frequency, row.mean_delta_f1, -row.mean_rank), reverse=True)

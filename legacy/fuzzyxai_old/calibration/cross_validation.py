from __future__ import annotations

from typing import Dict, Iterable, Sequence
import numpy as np

from .weight_calibration import calibrate_beta_weights, predict_disagreement, BETA_ORDER
from .dataset import CalibrationPair


def _mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def _spearman_like(y_true, y_pred) -> float:
    # small dependency-free Spearman approximation via rank correlation
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))
    if np.std(rt) == 0 or np.std(rp) == 0:
        return 0.0
    return float(np.corrcoef(rt, rp)[0, 1])


def cross_validate_beta(pairs: Sequence[CalibrationPair], *, folds: int = 5, seed: int = 42) -> Dict:
    if len(pairs) < folds:
        raise ValueError('number of pairs must be at least number of folds')
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    fold_ids = np.array_split(indices, folds)
    base_beta = {k: 1.0 / len(BETA_ORDER) for k in BETA_ORDER}
    rows = []
    for test_idx in fold_ids:
        train_idx = np.setdiff1d(indices, test_idx)
        train = [pairs[i] for i in train_idx]
        test = [pairs[i] for i in test_idx]
        beta = calibrate_beta_weights([p.features for p in train], [p.expert_score for p in train], include_reduction=True)
        y_true = [p.expert_score for p in test]
        y_base = [predict_disagreement(p.features, base_beta) for p in test]
        y_cal = [predict_disagreement(p.features, beta) for p in test]
        rows.append({
            'n_train': len(train),
            'n_test': len(test),
            'mse_base': _mse(y_true, y_base),
            'mse_calibrated': _mse(y_true, y_cal),
            'spearman_calibrated': _spearman_like(y_true, y_cal),
            'beta': beta,
        })
    return {
        'folds': rows,
        'mean_mse_base': float(np.mean([r['mse_base'] for r in rows])),
        'mean_mse_calibrated': float(np.mean([r['mse_calibrated'] for r in rows])),
        'mean_spearman_calibrated': float(np.mean([r['spearman_calibrated'] for r in rows])),
    }

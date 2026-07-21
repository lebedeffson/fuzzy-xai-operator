"""Deterministic split provenance and leakage audit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schemas import OperationKind, PartitionRole, SplitUseRecord


@dataclass(frozen=True)
class Q1Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    hashes: dict[str, str]


def make_split(labels: Sequence[int], *, seed: int = 4201) -> Q1Split:
    from sklearn.model_selection import train_test_split

    label_array = np.asarray(labels)
    indices = np.arange(len(label_array))
    train, remainder = train_test_split(indices, test_size=0.4, stratify=label_array, random_state=seed)
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        stratify=label_array[remainder],
        random_state=seed,
    )
    arrays = {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
    hashes = {key: hashlib.sha256(value.astype(np.int64).tobytes()).hexdigest() for key, value in arrays.items()}
    return Q1Split(arrays["train"], arrays["validation"], arrays["test"], hashes)


def standard_split_ledger(split: Q1Split) -> tuple[SplitUseRecord, ...]:
    return (
        SplitUseRecord("model-fit", OperationKind.FIT, (PartitionRole.TRAIN,), split.hashes),
        SplitUseRecord(
            "policy-calibration",
            OperationKind.CALIBRATE,
            (PartitionRole.TRAIN, PartitionRole.VALIDATION),
            split.hashes,
        ),
        SplitUseRecord("final-test", OperationKind.EVALUATE, (PartitionRole.TEST,), split.hashes),
    )

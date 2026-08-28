from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ABNORMAL = frozenset({"MI", "STTC", "CD", "HYP"})


def canonical_lead_name(value: str) -> str:
    """Normalize the capitalization used by WFDB headers without reordering leads."""
    normalized = str(value).strip().upper()
    aliases = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class ECGRecord:
    ecg_id: int
    patient_id: int
    fold: int
    filename_lr: str
    diagnostic_superclasses: tuple[str, ...]
    included: bool
    label: int | None
    exclusion_reason: str | None


def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return database, statements


def diagnostic_class_map(statements: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for code, row in statements.iterrows():
        if float(row.get("diagnostic", 0) or 0) == 1 and pd.notna(row.get("diagnostic_class")):
            mapping[str(code)] = str(row["diagnostic_class"])
    return mapping


def derive_primary_records(database: pd.DataFrame, statements: pd.DataFrame) -> list[ECGRecord]:
    class_map = diagnostic_class_map(statements)
    result = []
    for ecg_id, row in database.iterrows():
        codes = ast.literal_eval(str(row["scp_codes"]))
        classes = tuple(sorted({class_map[code] for code in codes if code in class_map}))
        class_set = set(classes)
        if not classes:
            included, label, reason = False, None, "no_diagnostic_superclass"
        elif "NORM" in class_set and class_set & ABNORMAL:
            included, label, reason = False, None, "ambiguous_norm_and_abnormal"
        elif class_set == {"NORM"}:
            included, label, reason = True, 0, None
        elif class_set & ABNORMAL:
            included, label, reason = True, 1, None
        else:
            included, label, reason = False, None, "outside_primary_binary_policy"
        result.append(ECGRecord(int(ecg_id), int(row["patient_id"]), int(row["strat_fold"]), str(row["filename_lr"]), classes, included, label, reason))
    return result


def split_name(fold: int) -> str:
    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    raise ValueError(f"unexpected PTB-XL strat_fold: {fold}")


def assert_patient_disjoint(records: list[ECGRecord]) -> None:
    patients: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    for record in records:
        if record.included:
            patients[split_name(record.fold)].add(record.patient_id)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = patients[left] & patients[right]
        if overlap:
            raise ValueError(f"patient leakage between {left}/{right}: {sorted(overlap)[:5]}")


def load_waveform(root: Path, record: ECGRecord) -> np.ndarray:
    signal, fields = wfdb.rdsamp(str(root / record.filename_lr))
    observed_leads = tuple(canonical_lead_name(name) for name in fields["sig_name"])
    if observed_leads != LEADS:
        raise ValueError(f"unexpected lead order for ecg_id={record.ecg_id}: {fields['sig_name']}")
    value = np.asarray(signal, dtype=np.float32).T
    if value.shape != (12, 1000) or not np.isfinite(value).all():
        raise ValueError(f"invalid records100 signal for ecg_id={record.ecg_id}: {value.shape}")
    return value


def record_dict(record: ECGRecord) -> dict[str, Any]:
    return {"ecg_id": record.ecg_id, "patient_id": record.patient_id, "strat_fold": record.fold, "split": split_name(record.fold), "filename_lr": record.filename_lr, "diagnostic_superclasses": list(record.diagnostic_superclasses), "included": record.included, "binary_label": record.label, "exclusion_reason": record.exclusion_reason}

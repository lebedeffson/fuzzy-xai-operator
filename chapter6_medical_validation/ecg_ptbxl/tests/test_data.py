from __future__ import annotations

import pandas as pd

from chapter6_medical_validation.ecg_ptbxl.src.data import ECGRecord, assert_patient_disjoint, canonical_lead_name, derive_primary_records


def test_wfdb_augmented_limb_lead_capitalization_is_normalized_without_reordering():
    assert tuple(canonical_lead_name(name) for name in ("I", "II", "III", "AVR", "AVL", "AVF", "V1")) == (
        "I", "II", "III", "aVR", "aVL", "aVF", "V1"
    )


def test_primary_labels_are_deterministic_and_ambiguous_is_explicit():
    statements = pd.DataFrame({"diagnostic": [1, 1, 1], "diagnostic_class": ["NORM", "MI", "STTC"]}, index=["NORM", "MI", "STTC"])
    database = pd.DataFrame({"patient_id": [1, 2, 3, 4], "strat_fold": [1, 9, 10, 1], "filename_lr": ["a", "b", "c", "d"], "scp_codes": ["{'NORM': 100}", "{'MI': 100}", "{'NORM': 50, 'STTC': 50}", "{'OTHER': 100}"]}, index=pd.Index([1, 2, 3, 4], name="ecg_id"))
    records = derive_primary_records(database, statements)
    assert [(row.included, row.label, row.exclusion_reason) for row in records] == [(True, 0, None), (True, 1, None), (False, None, "ambiguous_norm_and_abnormal"), (False, None, "no_diagnostic_superclass")]


def test_patient_overlap_across_official_folds_is_rejected():
    rows = [ECGRecord(1, 10, 1, "a", ("NORM",), True, 0, None), ECGRecord(2, 10, 10, "b", ("MI",), True, 1, None)]
    try:
        assert_patient_disjoint(rows)
    except ValueError as exc:
        assert "patient leakage" in str(exc)
    else:
        raise AssertionError("patient leakage was accepted")

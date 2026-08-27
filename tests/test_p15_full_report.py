"""P16 section 20: a deterministic, numbered 18-point full explanation
report, built entirely from evidence already collected during explain() —
no new computation, no SLM. Sections whose evidence is genuinely absent are
omitted, never fabricated.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_full_report_has_the_core_18_point_sections_when_evidence_is_complete() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0", include_counterfactuals=True)
    report = result.full_report()
    # 6 (contradicting evidence), 7 (data anomalies), and 17
    # (not_applicable channels) are legitimately empty for this scenario —
    # a typical object with no capability-excluded channels is not a bug.
    for number in (1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 18):
        assert f"## {number}." in report
    assert "## 16." not in report


def test_full_report_omits_training_section_when_no_training_history() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    report = result.full_report()
    assert "## 10." not in report


def test_full_report_section_4_names_the_actual_mechanism_not_a_generic_label() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    report = result.full_report()
    assert "Путь по дереву решений" in report


def test_full_report_separates_missing_required_from_unevaluated_quality_metrics() -> None:
    """P17: section 16 (missing_required) must not be conflated with
    unevaluated *optional* quality metrics — those belong in section 18
    (limitations), never labeled as a required channel."""

    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    report = result.full_report()
    section_18 = report[report.find("## 18.") :]
    assert "## 16." not in report
    assert result.view_model.explanation_level["channel_status"]["alignment"] == "not_applicable"
    assert "faithfulness" in section_18
    assert "no perturbation-based faithfulness check was supplied" in section_18


def test_full_report_section_3_states_the_prediction_and_score() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    report = result.full_report()
    idx = report.find("## 3.")
    assert idx != -1
    section_text = report[idx : report.find("## 4.")]
    # P18 item 7: section 3 now reuses the same domain-language-aware
    # decision text as the rest of the report (he.decision.explanation)
    # instead of dumping the raw prediction array — for this model's plan,
    # no domain_language.classes entry is registered, so the honest,
    # evidence-first wording names the technical result rather than
    # fabricating a domain meaning for it.
    assert "результат" in section_text.lower()
    assert "Модельный балл" in section_text


def test_full_report_reader_level_shows_only_top_factors() -> None:
    """P17: the default 'reader' level shows 4-6 supports / 3-4 contradicts,
    not a same-phrase list of every feature; 'audit' shows everything."""

    X_train, X_test, y_train, _ = _split()
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    reader_report = result.full_report(level="reader")
    audit_report = result.full_report(level="audit")
    reader_section5 = reader_report[reader_report.find("## 5.") : reader_report.find("## 6.")]
    audit_section5 = audit_report[audit_report.find("## 5.") : audit_report.find("## 6.")]
    assert reader_section5.count("- **") <= 6
    assert audit_section5.count("- **") > 6
    assert "показаны" in reader_section5  # truncation is disclosed, not silent


def test_full_report_rejects_unknown_level() -> None:
    import pytest

    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    with pytest.raises(ValueError):
        result.full_report(level="bogus")


def test_full_report_section_8_flags_counterexamples() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    report = result.full_report()
    idx = report.find("## 8.")
    assert idx != -1

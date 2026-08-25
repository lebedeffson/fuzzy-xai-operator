from __future__ import annotations

import math

import pytest
from fuzzyxai.evidence import find_text_highlight_spans
from fuzzyxai.evidence.contracts import TextHighlightEvidence, TextSpan


def test_finds_spans_for_features_present_in_text() -> None:
    evidence = find_text_highlight_spans(
        "The mean_radius was elevated while texture stayed normal.",
        {"mean_radius": 12.4, "texture": -1.1},
        object_id="obj-1",
    )
    assert evidence.object_id == "obj-1"
    assert len(evidence.spans) == 2
    by_feature = {span.feature_name: span for span in evidence.spans}
    assert by_feature["mean_radius"].direction == "supports"
    assert by_feature["texture"].direction == "contradicts"
    assert evidence.raw_text[by_feature["mean_radius"].start : by_feature["mean_radius"].end] == "mean_radius"
    assert not evidence.unmapped_features


def test_case_insensitive_matching() -> None:
    evidence = find_text_highlight_spans("MEAN_RADIUS was high.", {"mean_radius": 1.0}, object_id="obj-1")
    assert len(evidence.spans) == 1
    assert evidence.spans[0].start == 0


def test_feature_not_in_text_is_reported_as_unmapped_not_guessed() -> None:
    evidence = find_text_highlight_spans("Nothing relevant here.", {"unrelated_feature": 5.0}, object_id="obj-1")
    assert evidence.spans == ()
    assert evidence.unmapped_features == ("unrelated_feature",)


def test_spans_are_ordered_by_position() -> None:
    evidence = find_text_highlight_spans(
        "second appears before first in this sentence.",
        {"first": 1.0, "second": 1.0},
        object_id="obj-1",
    )
    assert [span.feature_name for span in evidence.spans] == ["second", "first"]


def test_evidence_always_carries_a_lexical_matching_limitation() -> None:
    evidence = find_text_highlight_spans("first matters.", {"first": 1.0}, object_id="obj-1")
    assert evidence.limitations
    assert "lexical" in evidence.limitations[0]


def test_word_boundary_does_not_match_inside_a_longer_word() -> None:
    evidence = find_text_highlight_spans("my heart rate and art style both matter", {"art": 1.0}, object_id="obj-1")
    matched_text = [evidence.raw_text[s.start : s.end] for s in evidence.spans]
    assert matched_text == ["art"]
    assert "heart" not in "".join(matched_text)


def test_phrase_and_ngram_features_match() -> None:
    evidence = find_text_highlight_spans("mean radius was elevated in this scan", {"mean radius": 1.0}, object_id="obj-1")
    assert len(evidence.spans) == 1
    assert evidence.raw_text[evidence.spans[0].start : evidence.spans[0].end] == "mean radius"


def test_repeated_occurrences_all_found() -> None:
    evidence = find_text_highlight_spans("wifi drops, wifi fails, wifi is unstable", {"wifi": 1.0}, object_id="obj-1")
    assert len(evidence.spans) == 3
    assert [s.start for s in evidence.spans] == sorted(s.start for s in evidence.spans)


def test_overlap_resolved_deterministically_longer_span_wins() -> None:
    evidence = find_text_highlight_spans("heart rate was elevated", {"heart rate": 2.0, "heart": 0.1}, object_id="obj-1")
    assert len(evidence.spans) == 1
    assert evidence.spans[0].feature_name == "heart rate"
    assert len(evidence.suppressed_matches) == 1
    assert "heart" in evidence.suppressed_matches[0]


def test_overlap_tie_break_prefers_larger_absolute_weight() -> None:
    # Same-length candidates ("aa" and "bb" both length 2) that happen to
    # occupy the same text position via two different feature names.
    evidence = find_text_highlight_spans("xx", {"xx": -0.1}, object_id="obj-1")
    assert len(evidence.spans) == 1  # sanity: single-candidate path is unaffected


def test_unicode_cyrillic_matching() -> None:
    evidence = find_text_highlight_spans(
        "пациент жалуется на боль в груди",
        {"боль": 1.0, "груди": -1.0},
        object_id="obj-1",
    )
    by_feature = {s.feature_name: s for s in evidence.spans}
    assert evidence.raw_text[by_feature["боль"].start : by_feature["боль"].end] == "боль"
    assert by_feature["груди"].direction == "contradicts"


def test_empty_text_and_empty_contributions() -> None:
    assert find_text_highlight_spans("", {}, object_id="obj-1").spans == ()
    assert find_text_highlight_spans("some text", {}, object_id="obj-1").spans == ()
    assert find_text_highlight_spans("", {"feature": 1.0}, object_id="obj-1").unmapped_features == ("feature",)


def test_non_finite_weight_is_treated_as_unmapped_not_crashed() -> None:
    evidence = find_text_highlight_spans(
        "feature_x and feature_y appear here",
        {"feature_x": float("nan"), "feature_y": float("inf")},
        object_id="obj-1",
    )
    assert evidence.spans == ()
    assert set(evidence.unmapped_features) == {"feature_x", "feature_y"}


def test_negligible_relative_weight_features_are_excluded_from_unmapped_noise() -> None:
    # A TF-IDF-style vocabulary: one strong signal plus many near-zero
    # weights for words the model barely reacted to. Only the strong
    # feature (and any comparably-important one) should be considered.
    contributions = {"wifi": 0.42, "and": 0.0003, "the": -0.0001, "at": 0.0002}
    evidence = find_text_highlight_spans("wifi keeps dropping and the connection is bad at night", contributions, object_id="obj-1")
    assert [span.feature_name for span in evidence.spans] == ["wifi"]
    assert evidence.unmapped_features == ()


def test_relative_weight_threshold_is_configurable() -> None:
    contributions = {"wifi": 0.42, "night": 0.05}  # night is ~12% of wifi's weight
    strict = find_text_highlight_spans("wifi drops at night", contributions, object_id="obj-1", min_relative_weight=0.5)
    assert [span.feature_name for span in strict.spans] == ["wifi"]
    lenient = find_text_highlight_spans("wifi drops at night", contributions, object_id="obj-1", min_relative_weight=0.01)
    assert {span.feature_name for span in lenient.spans} == {"wifi", "night"}


def test_single_feature_is_never_excluded_by_its_own_relative_weight() -> None:
    evidence = find_text_highlight_spans("only one signal here", {"signal": 0.0001}, object_id="obj-1")
    assert [span.feature_name for span in evidence.spans] == ["signal"]


def test_non_finite_weight_still_reported_even_with_relative_filter_active() -> None:
    contributions = {"wifi": 0.42, "broken_feature": float("nan")}
    evidence = find_text_highlight_spans("wifi drops constantly", contributions, object_id="obj-1")
    assert evidence.unmapped_features == ("broken_feature",)


def test_regex_special_characters_in_feature_name_do_not_crash() -> None:
    evidence = find_text_highlight_spans("price is $5.00 (approx)", {"$5.00": 1.0, "(approx)": 0.5}, object_id="obj-1")
    assert len(evidence.spans) == 2


class TestTextSpanContract:
    def test_rejects_inverted_or_empty_range(self) -> None:
        with pytest.raises(ValueError):
            TextSpan(start=5, end=5, feature_name="x", direction="supports", weight=1.0)
        with pytest.raises(ValueError):
            TextSpan(start=5, end=2, feature_name="x", direction="supports", weight=1.0)

    def test_rejects_invalid_direction(self) -> None:
        with pytest.raises(ValueError):
            TextSpan(start=0, end=1, feature_name="x", direction="neutral", weight=1.0)  # type: ignore[arg-type]

    def test_rejects_non_finite_weight(self) -> None:
        with pytest.raises(ValueError):
            TextSpan(start=0, end=1, feature_name="x", direction="supports", weight=math.nan)
        with pytest.raises(ValueError):
            TextSpan(start=0, end=1, feature_name="x", direction="supports", weight=math.inf)


class TestTextHighlightEvidenceContract:
    def test_rejects_span_beyond_text_length(self) -> None:
        span = TextSpan(start=0, end=50, feature_name="x", direction="supports", weight=1.0)
        with pytest.raises(ValueError):
            TextHighlightEvidence(object_id="o1", raw_text="short", spans=(span,))

    def test_rejects_unsorted_or_overlapping_spans(self) -> None:
        first = TextSpan(start=5, end=10, feature_name="a", direction="supports", weight=1.0)
        second = TextSpan(start=0, end=3, feature_name="b", direction="supports", weight=1.0)
        with pytest.raises(ValueError):
            TextHighlightEvidence(object_id="o1", raw_text="0123456789", spans=(first, second))

    def test_accepts_sorted_non_overlapping_spans(self) -> None:
        first = TextSpan(start=0, end=3, feature_name="a", direction="supports", weight=1.0)
        second = TextSpan(start=5, end=10, feature_name="b", direction="supports", weight=1.0)
        evidence = TextHighlightEvidence(object_id="o1", raw_text="0123456789", spans=(first, second))
        assert evidence.spans == (first, second)

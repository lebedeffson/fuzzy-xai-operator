from __future__ import annotations

from fuzzyxai.evidence.contracts import TextHighlightEvidence, TextSpan
from fuzzyxai.visualization.text_highlight import render_text_highlight_html


def test_raw_text_and_feature_names_are_html_escaped() -> None:
    raw = '<script>alert(1)</script> feature_x looks bad & "dangerous"'
    start = raw.index("feature_x")
    span = TextSpan(start=start, end=start + len("feature_x"), feature_name='<b>feature_x</b>"', direction="supports", weight=1.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text=raw, spans=(span,))
    html = render_text_highlight_html(evidence)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>feature_x</b>" not in html  # feature name markup must not leak into the DOM


def test_empty_spans_show_text_and_explicit_message_not_empty_view() -> None:
    evidence = TextHighlightEvidence(object_id="o1", raw_text="no evidence text here", spans=())
    html = render_text_highlight_html(evidence)
    assert "no evidence text here" in html
    assert "не удалось сопоставить" in html


def test_empty_raw_text_still_renders_a_non_empty_view() -> None:
    evidence = TextHighlightEvidence(object_id="o1", raw_text="", spans=())
    html = render_text_highlight_html(evidence)
    assert html.strip()


def test_long_text_is_excerpted_without_cutting_a_span() -> None:
    long_text = ("padding " * 200) + "important_feature is the key phrase here" + (" more padding" * 200)
    start = long_text.index("important_feature")
    span = TextSpan(start=start, end=start + len("important_feature"), feature_name="important_feature", direction="supports", weight=2.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text=long_text, spans=(span,))
    html = render_text_highlight_html(evidence, max_chars=100, context_chars=20)
    assert len(html) < len(long_text)
    assert "important_feature" in html
    assert "…" in html or "&hellip;" in html


def test_full_text_shown_when_under_max_chars() -> None:
    text = "short measured text with feature_x present"
    start = text.index("feature_x")
    span = TextSpan(start=start, end=start + len("feature_x"), feature_name="feature_x", direction="supports", weight=1.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text=text, spans=(span,))
    html = render_text_highlight_html(evidence, max_chars=1000)
    assert "short measured text with" in html
    assert "present" in html


def test_supports_and_contradicts_are_visually_distinct_beyond_color() -> None:
    supports = TextSpan(start=0, end=1, feature_name="a", direction="supports", weight=1.0)
    contradicts = TextSpan(start=2, end=3, feature_name="b", direction="contradicts", weight=-1.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text="a b c", spans=(supports, contradicts))
    html = render_text_highlight_html(evidence)
    # distinguished by icon (not color alone) so grayscale/colorblind readers can tell them apart
    assert "▲" in html
    assert "▼" in html
    assert "solid" in html and "dashed" in html


def test_legend_present_when_spans_exist() -> None:
    span = TextSpan(start=0, end=1, feature_name="a", direction="supports", weight=1.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text="a b c", spans=(span,))
    html = render_text_highlight_html(evidence)
    assert "legend" in html
    assert "поддерживает" in html


def test_whitespace_and_newlines_preserved_via_pre_wrap() -> None:
    text = "line one\nline two with feature_x here"
    start = text.index("feature_x")
    span = TextSpan(start=start, end=start + len("feature_x"), feature_name="feature_x", direction="supports", weight=1.0)
    evidence = TextHighlightEvidence(object_id="o1", raw_text=text, spans=(span,))
    html = render_text_highlight_html(evidence)
    assert "white-space:pre-wrap" in html


def test_unmapped_features_shown_as_footnote() -> None:
    evidence = TextHighlightEvidence(object_id="o1", raw_text="text with a matched word", spans=(), unmapped_features=("ghost_feature",))
    html = render_text_highlight_html(evidence)
    assert "ghost_feature" in html

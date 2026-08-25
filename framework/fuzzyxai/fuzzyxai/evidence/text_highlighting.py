from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .contracts import TextHighlightEvidence, TextSpan

_WORD_ONLY = re.compile(r"^\w+$", re.UNICODE)


def _pattern_for(feature_name: str) -> re.Pattern[str] | None:
    needle = feature_name.strip()
    if not needle:
        return None
    escaped = re.escape(needle)
    # Word-boundary matching only makes sense when the feature name is made
    # entirely of word characters (letters/digits/underscore) — otherwise
    # \b around punctuation is meaningless and we fall back to a literal
    # substring search. This is what keeps "art" from matching inside
    # "heart" while still letting phrases and hyphenated names match.
    if _WORD_ONLY.match(needle):
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE | re.UNICODE)
    return re.compile(escaped, re.IGNORECASE | re.UNICODE)


def find_text_highlight_spans(
    raw_text: str,
    feature_contributions: Mapping[str, float],
    *,
    object_id: str,
    min_relative_weight: float = 0.05,
) -> TextHighlightEvidence:
    """Locate every measured feature contribution inside the raw text it came from.

    Purely lexical (case-insensitive substring/word-boundary search, with
    phrase/n-gram support) — no semantic or embedding-based attribution is
    claimed. A feature name that does not occur as a literal token anywhere
    in ``raw_text`` is reported in ``unmapped_features`` rather than guessed
    at. When two features' occurrences overlap in the text, the conflict is
    resolved deterministically (longer match wins, then larger |weight|,
    then feature name) and the losing match is recorded in
    ``suppressed_matches`` rather than silently dropped.

    ``min_relative_weight`` excludes features whose |contribution| is below
    that fraction of the largest |contribution| among the supplied features
    (default 5%) from consideration entirely — for a text model with a large
    vocabulary, most features are absent from any single input and carry a
    near-zero contribution; without this filter, `unmapped_features` fills up
    with a wall of vocabulary noise no reader asked about, instead of the
    handful of features that actually mattered and simply weren't found in
    the text. A non-finite contribution is a genuine data problem and is
    still always reported in `unmapped_features`, never silently dropped.
    """

    candidates: list[TextSpan] = []
    unmapped: list[str] = []
    finite_abs_weights = [abs(float(weight)) for weight in feature_contributions.values() if math.isfinite(float(weight))]
    max_abs_weight = max(finite_abs_weights, default=0.0)
    weight_threshold = max_abs_weight * min_relative_weight
    for feature_name, weight in feature_contributions.items():
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight):
            # A non-finite contribution has no honest direction/magnitude to
            # display; treat it the same as "not mappable" rather than
            # crashing the whole evidence collection over one bad value.
            unmapped.append(str(feature_name))
            continue
        if max_abs_weight > 0 and abs(numeric_weight) < weight_threshold:
            # Negligible relative to the strongest signal for this object —
            # not interesting enough to report as evidence or as an
            # unmapped-feature notice.
            continue
        pattern = _pattern_for(str(feature_name))
        if pattern is None:
            continue
        found = list(pattern.finditer(raw_text))
        if not found:
            unmapped.append(str(feature_name))
            continue
        direction = "supports" if numeric_weight >= 0 else "contradicts"
        for match in found:
            candidates.append(TextSpan(match.start(), match.end(), str(feature_name), direction, numeric_weight))

    # Deterministic priority: longer span first, then larger |weight|, then
    # feature name — ties broken the same way on every run, regardless of
    # dict iteration order.
    candidates.sort(key=lambda span: (-(span.end - span.start), -abs(span.weight), span.feature_name, span.start))

    selected: list[TextSpan] = []
    occupied: list[tuple[int, int]] = []
    suppressed: list[str] = []
    for span in candidates:
        overlap = next(((start, end) for start, end in occupied if span.start < end and start < span.end), None)
        if overlap is not None:
            suppressed.append(
                f"{span.feature_name}@{span.start}-{span.end} suppressed by overlap with an existing match at {overlap[0]}-{overlap[1]}"
            )
            continue
        selected.append(span)
        occupied.append((span.start, span.end))

    selected.sort(key=lambda span: span.start)
    return TextHighlightEvidence(
        object_id=str(object_id),
        raw_text=raw_text,
        spans=tuple(selected),
        unmapped_features=tuple(sorted(set(unmapped))),
        suppressed_matches=tuple(suppressed),
    )

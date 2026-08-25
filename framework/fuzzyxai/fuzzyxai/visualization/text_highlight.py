from __future__ import annotations

from html import escape

from fuzzyxai.evidence.contracts import TextHighlightEvidence, TextSpan

# Colors are paired with a non-color signal (icon + underline style) so the
# distinction survives grayscale printing and color-vision deficiency, not
# only a color legend.
_SUPPORTS_COLOR = "#c9ecd8"
_SUPPORTS_BORDER = "#1f7a4d"
_CONTRADICTS_COLOR = "#f6cccc"
_CONTRADICTS_BORDER = "#a23b3b"
_SUPPORTS_ICON = "▲"  # ▲
_CONTRADICTS_ICON = "▼"  # ▼

DEFAULT_MAX_CHARS = 600
DEFAULT_CONTEXT_CHARS = 80

_NO_EVIDENCE_MESSAGE = "Вклад признаков не удалось сопоставить с исходным текстом."


def _mark(text: str, span: TextSpan) -> str:
    supports = span.direction == "supports"
    color = _SUPPORTS_COLOR if supports else _CONTRADICTS_COLOR
    border = _SUPPORTS_BORDER if supports else _CONTRADICTS_BORDER
    icon = _SUPPORTS_ICON if supports else _CONTRADICTS_ICON
    style = f"background-color:{color};border-bottom:2px {'solid' if supports else 'dashed'} {border};padding:0 1px"
    title = escape(f"{span.feature_name}: {span.weight:+.4f} ({span.direction})", quote=True)
    return f'<mark style="{style}" title="{title}">{escape(text)}<sup aria-hidden="true">{icon}</sup></mark>'


def _legend() -> str:
    supports = f'<mark style="background-color:{_SUPPORTS_COLOR};border-bottom:2px solid {_SUPPORTS_BORDER};padding:0 4px">{_SUPPORTS_ICON} поддерживает</mark>'
    contradicts = f'<mark style="background-color:{_CONTRADICTS_COLOR};border-bottom:2px dashed {_CONTRADICTS_BORDER};padding:0 4px">{_CONTRADICTS_ICON} противоречит</mark>'
    return f'<div class="fuzzyxai-text-highlight-legend" style="margin-top:6px;font-size:0.85em">{supports} &nbsp; {contradicts}</div>'


def _excerpt_windows(text_length: int, spans: tuple[TextSpan, ...], context_chars: int) -> list[tuple[int, int]]:
    """Merge context windows around every span into non-overlapping ranges."""

    windows = sorted((max(0, span.start - context_chars), min(text_length, span.end + context_chars)) for span in spans)
    merged: list[list[int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def render_text_highlight_html(
    evidence: TextHighlightEvidence,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> str:
    """Render raw text with measured feature spans marked, in reading order.

    Never introduces text that was not in ``evidence.raw_text``; span
    boundaries come directly from ``evidence.spans``, which were located by
    lexical matching only (see ``evidence.text_highlighting``). The text is
    HTML-escaped in full before any markup is layered on top, so neither the
    raw text nor a feature name can inject markup. Long text is shown as
    excerpts around each span (never cutting a span in half) rather than one
    unbounded block; if there are no spans, the plain (escaped) text is still
    shown along with an explicit statement that no evidence could be located
    — the view is never silently empty.
    """

    text = evidence.raw_text
    spans = tuple(sorted(evidence.spans, key=lambda span: span.start))
    style = 'style="white-space:pre-wrap;word-break:break-word;font-family:Georgia,serif;line-height:1.5"'

    def render_range(start: int, end: int) -> str:
        pieces: list[str] = []
        cursor = start
        for span in spans:
            if span.start < cursor or span.end > end:
                continue
            pieces.append(escape(text[cursor : span.start]))
            pieces.append(_mark(text[span.start : span.end], span))
            cursor = span.end
        pieces.append(escape(text[cursor:end]))
        return "".join(pieces)

    unmapped = evidence.unmapped_features
    footnote = ""
    if unmapped:
        footnote = f'<p style="color:#7b8790;font-size:0.85em">Не найдено в тексте буквально: {escape(", ".join(unmapped))}</p>'

    if not spans:
        body = escape(text) if len(text) <= max_chars else escape(text[:max_chars]) + "…"
        note = f'<p style="color:#7b8790;font-size:0.85em">{escape(_NO_EVIDENCE_MESSAGE)}</p>'
        return f'<div class="fuzzyxai-text-highlight" {style}>{body}</div>{note}{footnote}'

    if len(text) <= max_chars:
        body = render_range(0, len(text))
    else:
        windows = _excerpt_windows(len(text), spans, context_chars)
        rendered_windows = [render_range(start, end) for start, end in windows]
        separator = ' <span style="color:#9aa8af">…</span> '
        body = separator.join(rendered_windows)
        if windows[0][0] > 0:
            body = '<span style="color:#9aa8af">…</span> ' + body
        if windows[-1][1] < len(text):
            body = body + ' <span style="color:#9aa8af">…</span>'

    return f'<div class="fuzzyxai-text-highlight" {style}>{body}</div>{_legend()}{footnote}'

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_RAW_TEXT_REDACTED = "[raw text omitted by default; pass include_raw=True to include it]"
_RAW_IMAGE_REDACTED = "[raw image bytes omitted by default; pass include_raw=True to include them]"


def _redact_raw_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip the full raw object payload (text or image bytes) from an exported/serialized payload by default.

    Structured evidence (spans, offsets, tabular rows, region masks/bounding
    boxes) is not raw content and is left untouched — only the fields that
    carry the complete original text or image bytes are redacted, so a
    caller who forgets to check ``include_raw`` doesn't silently ship a
    user's raw document/image out through a log or export. Integrity
    references (e.g. the image's sha256) are kept even when redacted — a
    hash is not the content itself.
    """

    layers = payload.get("layers")
    if isinstance(layers, dict):
        highlights = layers.get("text_highlights")
        if isinstance(highlights, list):
            layers["text_highlights"] = [
                {**item, "raw_text": _RAW_TEXT_REDACTED} if isinstance(item, dict) else item for item in highlights
            ]
        images = layers.get("image_representations")
        if isinstance(images, list):
            layers["image_representations"] = [
                {**item, "image_png_base64": _RAW_IMAGE_REDACTED} if isinstance(item, dict) else item for item in images
            ]
    visual_spec = payload.get("visual_spec")
    if isinstance(visual_spec, dict):
        representation = visual_spec.get("object_representation")
        if isinstance(representation, dict) and representation.get("modality") == "text":
            visual_spec["object_representation"] = {
                **representation,
                "raw_excerpt": _RAW_TEXT_REDACTED,
                "highlighted_html": _RAW_TEXT_REDACTED,
            }
        elif isinstance(representation, dict) and representation.get("modality") == "image":
            visual_spec["object_representation"] = {
                **representation,
                "image_png_base64": _RAW_IMAGE_REDACTED,
            }
    return payload


@dataclass(frozen=True)
class ExplanationViewModel:
    """Backend-neutral contract shared by Python, web, and MATLAB views."""

    model: Mapping[str, Any]
    fuzzy: Mapping[str, Any]
    route: Sequence[Mapping[str, Any]]
    disagreement: Mapping[str, Any]
    risk: Mapping[str, Any]
    diagnostics: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    claims: Sequence[Mapping[str, Any]] | Mapping[str, Any] = field(default_factory=tuple)
    narrative: str = ""
    trace: Mapping[str, Any] = field(default_factory=dict)
    layers: Mapping[str, Any] = field(default_factory=dict)
    explanation_graph: Mapping[str, Any] = field(default_factory=dict)
    human_explanations: Mapping[str, Any] = field(default_factory=dict)
    quality_metrics: Mapping[str, float | None] = field(default_factory=dict)
    explanation_level: Mapping[str, Any] = field(default_factory=dict)
    visual_spec: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        return payload if include_raw else _redact_raw_text(payload)

    def to_json(self, *, indent: int = 2, include_raw: bool = False) -> str:
        return json.dumps(self.to_dict(include_raw=include_raw), ensure_ascii=False, indent=indent)

    def export_json(self, path: str | Path, *, include_raw: bool = False) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(include_raw=include_raw) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplanationViewModel:
        return cls(
            model=dict(payload.get("model", {})),
            fuzzy=dict(payload.get("fuzzy", {})),
            route=list(payload.get("route", [])),
            disagreement=dict(payload.get("disagreement", {})),
            risk=dict(payload.get("risk", {})),
            diagnostics=list(payload.get("diagnostics", [])),
            claims=(
                list(payload.get("claims", []))
                if isinstance(payload.get("claims", []), list)
                else dict(payload.get("claims", {}))
            ),
            narrative=str(payload.get("narrative", "")),
            trace=dict(payload.get("trace", {})),
            layers=dict(payload.get("layers", {})),
            explanation_graph=dict(payload.get("explanation_graph", {})),
            human_explanations=dict(payload.get("human_explanations", {})),
            quality_metrics=dict(payload.get("quality_metrics", {})),
            explanation_level=dict(payload.get("explanation_level", {})),
            visual_spec=dict(payload.get("visual_spec", {})),
            schema_version=str(payload.get("schema_version", "2.0")),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> ExplanationViewModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

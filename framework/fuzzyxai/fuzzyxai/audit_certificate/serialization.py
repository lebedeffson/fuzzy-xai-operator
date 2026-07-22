"""Canonical JSON serialization for audit certificates."""

from __future__ import annotations

import json
from dataclasses import asdict

from .certificate import ActionConditionedAuditCertificate


def certificate_to_json(certificate: ActionConditionedAuditCertificate, *, indent: int | None = None) -> str:
    return json.dumps(asdict(certificate), sort_keys=True, separators=None if indent else (",", ":"), indent=indent, ensure_ascii=True)

"""Structural certificate effect of removing a rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fuzzyxai.audit_certificate import ActionConditionedAuditCertificate


@dataclass(frozen=True)
class CertificateRuleEffect:
    estimand: str
    certificate_loss_rate: float
    mean_unsatisfied_contract_increase: float
    action_change_rate: float
    provenance_loss_rate: float
    n_objects: int


def certificate_effect(
    before: Sequence[ActionConditionedAuditCertificate],
    after: Sequence[ActionConditionedAuditCertificate],
    *,
    actions_before: Sequence[str] | None = None,
    actions_after: Sequence[str] | None = None,
) -> CertificateRuleEffect:
    if not before or len(before) != len(after):
        raise ValueError("before and after certificates must be non-empty and aligned")
    action_changes = 0
    if actions_before is not None or actions_after is not None:
        if actions_before is None or actions_after is None or len(actions_before) != len(before) or len(actions_after) != len(before):
            raise ValueError("action sequences must both be aligned")
        action_changes = sum(left != right for left, right in zip(actions_before, actions_after, strict=True))
    certificate_losses = sum(left.certificate_exists and not right.certificate_exists for left, right in zip(before, after, strict=True))
    increases = [max(0, len(right.unsatisfied_contracts) - len(left.unsatisfied_contracts)) for left, right in zip(before, after, strict=True)]
    provenance_losses = sum(
        sum(item.startswith("provenance:") for item in right.unsatisfied_contracts) > sum(item.startswith("provenance:") for item in left.unsatisfied_contracts)
        for left, right in zip(before, after, strict=True)
    )
    n = len(before)
    return CertificateRuleEffect("certificate", certificate_losses / n, sum(increases) / n, action_changes / n, provenance_losses / n, n)

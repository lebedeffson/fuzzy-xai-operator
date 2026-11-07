"""Batch certification of independent FuzzyXAI explanation view models.

Never averages structurally certified status into model accuracy; no training
or inference occurs here. By default, the summary stores no raw source cases.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .doctoral_certification import certify_payload


def certify_batch(
    objects: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    taxonomy: Mapping[str, Any],
    representation: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable aggregate plus per-object status and issue codes only.

    Cases must have externally assigned unique names. A bad case is reported
    as a validation error (fail closed); it cannot silently disappear from N.
    """
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for name, payload in objects:
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f'empty or duplicate batch identifier: {name!r}')
        seen.add(name)
        try:
            cert = certify_payload(payload, taxonomy=taxonomy, representation=representation, policy=policy)
            decision = cert['decision']
            records.append({
                'case_id': name, 'status': decision['status'], 'action': decision['action'],
                'auto_accept_permitted': bool(decision['auto_accept_permitted']),
                'n_issues': len(decision['issues']),
                'issue_codes': sorted(set(row['code'] for row in decision['issues'])),
                'receipt_sha256': cert['sha256'],
            })
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            records.append({
                'case_id': name, 'status': 'invalid', 'action': 'insufficient_evidence',
                'auto_accept_permitted': False, 'n_issues': 1,
                'issue_codes': ['D_unprocessable_case'],
                'receipt_sha256': None, 'error': str(error),
            })
    cases = sorted(records, key=lambda row: row['case_id'])
    statuses = Counter(str(row['status']) for row in cases)
    by_code = Counter(code for row in cases for code in row['issue_codes'])
    return {
        'schema_version': 'FuzzyXAI-DoctoralBatch/1.0',
        'scope': 'structural_route_compliance_not_prediction_accuracy',
        'n_cases': len(cases),
        'status_counts': dict(sorted(statuses.items())),
        'n_auto_accept_permitted': sum(row['auto_accept_permitted'] for row in cases),
        'issue_case_counts': dict(sorted(by_code.items())),
        'cases': cases,
        'not_validated': ['prediction accuracy', 'calibration', 'clinical utility', 'model generalization'],
    }

from __future__ import annotations


def validate_claim_status(status: str, sealed_scored: bool) -> None:
    if not sealed_scored and status in {"supported", "not_supported"}:
        raise ValueError("confirmatory status is forbidden before sealed scoring")


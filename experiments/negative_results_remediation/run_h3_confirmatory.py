from __future__ import annotations

from .common import ARTIFACTS, require_file, verify_protocol, write_json


def main() -> None:
    require_file(ARTIFACTS / "lock" / "negative_remediation_lock.json", "protocol must be frozen before confirmation")
    sealed = ARTIFACTS / "sealed" / "independent_confirmatory_manifest.json"
    output = ARTIFACTS / "h3" / "confirmatory_status.json"
    if not sealed.is_file():
        write_json(
            output,
            {
                "protocol_sha256": verify_protocol(),
                "status": "blocked_no_independent_sealed_dataset",
                "test_opened": False,
                "H3-R1": "not_evaluated",
                "H3-R2": "not_evaluated",
                "H3-R3": "not_evaluated",
                "reason": "No independent sealed dataset was registered before the remediation lock.",
                "positive_claim_allowed": False,
            },
        )
        print("BLOCKED remediation-h3-confirmatory test_opened=false reason=no_independent_sealed_dataset")
        return
    raise RuntimeError("sealed H3 runner requires a separately audited dataset adapter; refusing implicit test opening")


if __name__ == "__main__":
    main()

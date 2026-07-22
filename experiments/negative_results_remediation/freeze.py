from __future__ import annotations

from .common import ARTIFACTS, git_commit, read_json, sha256_file, verify_protocol, write_json


def main() -> None:
    summaries = []
    for iteration in ("r1", "r2", "r3"):
        path = ARTIFACTS / "formative" / iteration / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"formative stop rule incomplete: missing {path}")
        summaries.append({"iteration": iteration.upper(), "sha256": sha256_file(path), "summary": read_json(path)})
    lock = {
        "schema_version": "1.0",
        "protocol_sha256": verify_protocol(),
        "implementation_commit": git_commit(),
        "formative_iterations": summaries,
        "formative_iteration_count": 3,
        "maximum_formative_iterations": 3,
        "frozen_after_r3_regardless_of_result": True,
        "confirmatory_test_opened": False,
        "independent_confirmatory_dataset_registered": False,
        "h3_r1_r3_confirmatory_status": "blocked_no_independent_sealed_dataset",
        "post_lock_tuning_allowed": False,
    }
    write_json(ARTIFACTS / "lock" / "negative_remediation_lock.json", lock)
    print("PASS remediation-freeze iterations=3 independent_confirmatory=false")


if __name__ == "__main__":
    main()

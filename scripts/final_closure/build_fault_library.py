#!/usr/bin/env python3
from __future__ import annotations

from fuzzyxai.final_closure import compositional_faults, fault_library

from common import EVIDENCE, write


def main() -> None:
    templates, compositions = fault_library(), compositional_faults()
    payload = {
        "phase": "formative_library",
        "controlled_fault_templates": [item.__dict__ for item in templates],
        "compositional_templates": list(compositions),
        "terminology": "realistic replayed deployment incidents",
        "confirmatory_claim_allowed": False,
    }
    write(EVIDENCE / "fault_library.json", payload)
    print(f"PASS: final_fault_library templates={len(templates)} compositional={len(compositions)}")


if __name__ == "__main__":
    main()

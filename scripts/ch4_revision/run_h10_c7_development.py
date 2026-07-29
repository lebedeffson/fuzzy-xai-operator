#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import run_development_tournament
from fuzzyxai.experiments.h10_c7_model_lock import verify_model_weight_lock
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
    LocalTransformerCrossEncoder,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    HashingCodeEncoder,
    LocalTransformerCodeEncoder,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--model-weight-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-smoke-encoders", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.model_registry.read_text(encoding="utf-8"))
    encoders = []
    cross_encoder = None
    if not args.allow_smoke_encoders:
        verify_model_weight_lock(
            args.model_registry,
            args.model_weight_lock,
            require_read_only=True,
        )
        pending = [
            item.get("model_name")
            for group in ("dense_encoders", "cross_encoders")
            for item in registry.get(group, [])
            if item.get("weights_sha256")
            in {None, "", "PENDING_LOCAL_WEIGHT_VERIFICATION"}
        ]
        if pending:
            raise SystemExit(
                "H10-C7 model weights are not locally hash-verified: "
                + ", ".join(map(str, pending))
            )
    for item in registry.get("dense_encoders", []):
        if item.get("backend") == "local_transformer":
            encoders.append(
                LocalTransformerCodeEncoder(
                    str(item["model_name"]),
                    str(item["revision"]),
                )
            )
    cross_values = registry.get("cross_encoders", [])
    if cross_values and not args.allow_smoke_encoders:
        item = cross_values[0]
        cross_encoder = LocalTransformerCrossEncoder(
            str(item["model_name"]),
            str(item["revision"]),
        )
    if args.allow_smoke_encoders:
        encoders = (
            HashingCodeEncoder(256),
            HashingCodeEncoder(384),
        )
    if len(encoders) < 2:
        raise SystemExit(
            "H10-C7 development requires two registered dense encoders"
        )
    result = run_development_tournament(
        args.manifest,
        args.gold,
        args.output,
        args.root.resolve(),
        GuidedNaturalDiagnosisEngine(
            dense_encoders=encoders,
            cross_encoder=cross_encoder,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

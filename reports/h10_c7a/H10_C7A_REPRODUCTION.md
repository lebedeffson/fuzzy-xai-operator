# H10-C7A reproduction

## Inputs

- Base open replay: 30 H10-C5c incidents from 8 repositories.
- Development extension: the first 10 disclosed H10-C5b incidents ordered by
  the already registered `selection_rank_sha256`.
- Combined development set: 40 incidents from 16 repositories.
- Runtime collection was not repeated.
- Project dependencies were not reinstalled.
- Observable and Gold manifests are physically separate.

## Commands

```bash
make h10-c7a-prepare-development \
  H10_C7_REPLAY_BUNDLE=/path/to/h10-c7-open-replay-bundle \
  H10_C7A_H10_C5B_MANIFEST=/path/to/H10_C5B_HELD_OUT_SCORING_MANIFEST.jsonl \
  H10_C7A_DEVELOPMENT_BUNDLE=/path/to/h10-c7a-development-bundle

make h10-c7a-run-development \
  H10_C7A_DEVELOPMENT_BUNDLE=/path/to/h10-c7a-development-bundle \
  H10_C7A_OUTPUT=results/h10_c7a/development
```

Both commands run with the project `PYTHONPATH`. Development scoring also
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Dense-only baselines are
reported unavailable when their locked local weights are absent; this does
not block structural scoring.

## Fail-closed boundaries

- The frozen R5 top-10 and top-20 prefixes are checked against
  `R5C_PER_INCIDENT.jsonl`.
- The original 30-incident R5 metrics must reproduce exactly.
- Gold keys are recursively rejected from the observable manifest.
- Development locks are written only when every preregistered gate passes.
- The runner exits non-zero on a development gate failure.

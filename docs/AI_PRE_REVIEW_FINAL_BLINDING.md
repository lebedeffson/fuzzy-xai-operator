# AI pre-review final blinding boundary

## Status

The `60ed5697d4d607df59556ea82de63527905f0f4f` bundle is retained as a technical prototype only. It must not be scored because reviewer-visible records disclosed outcome, action and structural answer-key fields.

The replacement pipeline separates three boundaries:

1. `study/ai_pre_review_final/public_formative/` contains the frozen reviewer-visible formative input: 240 cases, 720 randomized variants and only assets referenced by those cases.
2. `study/ai_pre_review_final/private/` contains the encrypted method/outcome scoring key and is ignored by Git and excluded from every public archive.
3. The 120-case confirmatory pool is withheld until a real formative acceptance record passes `ai-final-lock-confirmatory`.

No AI score, human score, domain approval, comprehension result or expert-action result is present in this milestone.

## Reviewer-visible contract

The public record contains only:

- anonymized case and variant identifiers;
- modality, task and audience;
- displayed prediction and confidence;
- observable channel availability;
- human-readable measured reasons;
- direction, normalized magnitude, rank, stability and source agreement;
- evidence references and explicit limitations;
- candidate explanation and prospective action;
- complete claim-to-evidence links.

The record excludes true outcomes, correctness, original strata, expected actions, hidden rupture labels, answer-key annotations and method identity. `class_0`-style labels are rejected for `domain_user` records.

## Evidence by modality

- Tabular reasons include an anonymized observed value and reference percentile.
- Image reasons include a visible thumbnail, region identifier and bounding box.
- Text reasons include the actual anonymized phrase and character position.
- Time-series reasons include the actual interval and signal channel.

Attributions are model-behavior evidence, not causal or domain truth. Similarity, sensitivity and fidelity metrics retain their declared scale and limitation.

## Public formative archive

Build and verify:

```bash
make ai-final-check
```

Focused components:

```bash
make ai-final-blinding-audit
make ai-final-validate-evidence
make ai-final-claim-registry
make ai-final-archive
```

The resulting ZIP is written under `release_artifacts/ai_pre_review_final/`. It contains 240 formative cases and 12 batches. It does not contain confirmatory records, private paths or the encrypted scoring key.

## Confirmatory lock

`make ai-final-lock-confirmatory` fails closed until `study/ai_pre_review_final/formative_acceptance.json` contains hashes for at least two real formative review runs, zero critical defects, accepted median thresholds and an explicit authorization timestamp. The acceptance hash must match the frozen public formative packet.

After the lock, confirmatory inputs may be released once. They must not be rebuilt or edited after outcomes are opened.

## Claim boundary

The technical milestone supports only the statement that a leakage-audited, interpretable formative review input was built reproducibly. AI repeatability, AI-human agreement, domain-language approval, comprehension improvement and expert-action utility remain `open_external`. Stable release remains blocked.

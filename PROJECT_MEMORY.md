# Project Memory

- Date: 2026-07-19
- Branch: `feat/human-explanation-layer`
- Previous green baseline: `884409f4cb1b`
- Human Explanation implementation: `8b4a865c93b9`
- Candidate version: `1.2.0rc1`
- Public CI run: `29691191720`
- Release tag: not created

## Current Focus

The primary product is the installable FuzzyXAI research framework and its reproducible evidence-first explanation pipeline. The generated DubnaXAI website remains quarantined. The current milestone adds a human communication contract above the verified claim graph without weakening traceability.

## Canonical Public API

```python
fx = FuzzyXAI.wrap(model, adapter="auto", explain_plan=plan)
result = fx.explain_one(item, object_id="85", reference_data=X_train)

human = result.explain_for(audience="domain_user", language="ru")
result.summary(audience="domain_user", detail="short")
result.summary(audience="ml_engineer", detail="full")
result.inspect("claim:C004").provenance()
result.audit()
```

`ExplainPlan.domain_language` is the canonical mapping from feature names, class codes, and action codes to domain wording. E0-E5 states the available evidence depth. `domain_user`, `ml_engineer`, `researcher`, and `auditor` state how that evidence is communicated. These concepts must not be conflated.

## Implemented Boundary

- typed `HumanExplanation`, statement cards, `ExplanationDetails`, and `AudienceProfile`;
- decision, at most three reasons, at most two concerns, reliability, action, and at most one tested change for `domain_user`;
- grouping, deduplication, comparative wording, and evidence-aware claim ranking;
- separate supports, contradictions, and trust limitations;
- internal rule/subgroup/claim IDs, E0-E5, operator symbols, and raw action codes excluded from domain-user text;
- every visible card retains non-empty `claim_refs` and `evidence_refs`;
- user, engineer, researcher, and auditor profiles serialized in `ExplanationViewModel`;
- one-cycle compatibility aliases `user`, `expert`, and `audit`;
- feature contribution claims and contribution nodes in the explanation graph;
- similar-case support evaluated against the current prediction; counterexamples do not enter the support block;
- image-mask similarity described as geometric overlap, never diagnosis probability;
- object 85 human explanation reports the observed epoch-16 transition and hidden rare-subtype degradation;
- `HumanExplanation` JSON schema included in the wheel;
- operator manifest remains `30/30` and maps the human layer to tests and golden evidence.
- Chapter 4 evidence contains `30/30` operator rows, `12/12` figures, and `4/4` embedded human-explanation files.

## Validation

- local Python 3.14 regression: `308 passed`;
- strict Ruff gate: PASS;
- strict MyPy gate: PASS;
- `make framework-release-check`: PASS, release subset `26 passed`;
- operator manifest: `30/30`, PASS;
- deterministic golden rebuild and checksum comparison: PASS;
- Human Explanation verifier: object 85 cards, medical similarity semantics, cross-model evidence, PASS;
- wheel/sdist build as `1.2.0rc1`: PASS;
- isolated wheel import and `explain_for()` smoke: PASS;
- wheel content check includes runtime, human layer, and JSON schema: PASS;
- deterministic Chapter 4 evidence ZIP: PASS, SHA256 `6f6b5a444bd492734a74e8fdfcbbf1580b4e97cbc3b952f23a9cc63a00178786`;
- public GitHub Actions run `29691191720`: Python 3.11, Python 3.12, and Octave PASS for `8b4a865c93b9`;
- comprehension pilot: `planned_not_run`.

## Claim Boundary

- Do not claim demonstrated comprehensibility until the external pilot is complete.
- Do not claim support for literally every model; native Torch, Keras, and ONNX adapters are not implemented.
- Surrogate rules remain labeled `surrogate` and include fidelity limitations.
- Similarity names its representation and metric and is not a probability or causal statement.
- Missing evidence results in an explicit limitation or review, never an invented metric or sentence.
- Medical examples are controlled research-only fixtures, not clinical conclusions or certification evidence.
- E0-E5 describes evidence depth, not model quality, reliability, or audience.
- MATLAB/Octave renders canonical JSON but does not independently implement the Python operator core.

## Repository And Release Policy

- Keep the generated website quarantined.
- Build the analysis ZIP only from the committed Git index with `python scripts/build_framework_release.py`.
- Never package the dirty working tree.
- Do not create `v1.2.0-rc1` until the documented comprehension pilot passes and the release is approved for tagging.

## Next Step

Run the documented comprehension pilot with independent domain users and model integrators. Until its raw responses and scoring are archived, use this candidate as a machine-verified Human Explanation Layer, not as evidence of universal human comprehensibility.

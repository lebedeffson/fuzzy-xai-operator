# Project Memory

- Date: 2026-07-19
- Branch: `feat/explanation-experience-completion`
- Baseline commit: `83d3b0016e38`
- Implementation commits: `3b51ca5`, `4a820f4`, `0d8754b`
- Release packaging commits: `d1b75bc`, `a490c09`
- Release-status commit: `f34532b`
- MATLAB typed-status compatibility fix: `db7a174`
- Candidate version: `1.1.0rc3`
- Release tag: not created

## Current Focus

The primary product is the installable FuzzyXAI research framework and its reproducible evidence-first explanation pipeline. The DubnaXAI generated website remains quarantined. The current milestone turns evidence operators into a coherent and inspectable account of how data became a decision.

## Canonical Public API

```python
fx = FuzzyXAI.wrap(model, adapter="auto", explain_plan=plan)
result = fx.explain_one(item, object_id="85", reference_data=X_train)

result.overview()
result.story()
result.inspect("rule:R31")
result.audit()
result.visualize(view="training_trace", backend="matplotlib")
```

The canonical adapter contract is `ModelAdapter`. The canonical visualization namespace is `fuzzyxai.visualization`; `visual` and `viz` are compatibility-only namespaces and are not runtime dependencies of the new public flow.

## Implemented Boundary

- `ExplanationClaim` separates `evidence_status`, effect, and severity;
- `ExplanationGraph` validates evidence-to-claim-to-diagnostic-to-action reachability;
- E0-E5 disclose available, missing, native, and surrogate channels;
- typed `ExplanationVisualSpec` is the only renderer input;
- typed inspection covers claims, rules, concepts, objects, evidence, diagnostics, and actions;
- ten Matplotlib and ten Plotly views contain real data rather than placeholders;
- object 85 contains 12 checkpoints and a controlled epoch-16 forgetting event;
- rule ablation stores measured before/after train, validation, test, subgroup, critical-error, and calibration values;
- controlled model scenarios cover callable black-box, sklearn linear, decision tree, ANFIS, and training history;
- the medical research fixture uses generated images and masks, computed IoU, distinct embedding similarity, counterexamples, and explicit limitations;
- golden evidence is deterministic and machine-verified;
- the Chapter 4 package maps all 30 manifest constructions to tests and evidence and contains 12 figures.

## Validation

- local Python 3.14 regression: `304 passed`;
- clean committed snapshot on Python 3.12: `304 passed`;
- strict public-contract Ruff and MyPy gates: PASS;
- framework release gate: PASS;
- operator manifest: `30/30`, PASS;
- deterministic golden rebuild: PASS;
- wheel/sdist and isolated wheel import as `1.1.0rc3`: PASS;
- public CI run `29685440340`: Python 3.11, Python 3.12, and Octave PASS for `db7a174`;
- local Octave: unavailable; public Octave confirmation PASS;
- comprehension pilot: `planned_not_run`.

## Claim Boundary

- Do not claim support for literally every model; native Torch, Keras, and ONNX adapters are not implemented.
- Surrogate rules remain labeled `surrogate` and include fidelity limitations.
- Similarity always names representation and metric and is not a probability or causal statement.
- Missing evidence results in `insufficient_evidence` or review, never invented metrics.
- Medical examples are research-only and are not clinical or production certification evidence.
- E0-E5 describes evidence depth, not model quality or a universal interpretability score.
- MATLAB/Octave renders the canonical JSON but does not independently implement the Python operator core.

## Repository And Release Policy

- Keep the generated site quarantined until the framework API and explanation contracts are stable.
- Build the analysis ZIP only from the committed Git index with `python scripts/build_framework_release.py`.
- The full analysis ZIP retains tracked reports and fixtures required by the regression suite, while excluding the quarantined site and editor state.
- Do not create `v1.1.0-rc3` until public Python 3.11/3.12 and Octave CI are green and the external comprehension gate is complete.

## Next Step

Run the documented comprehension pilot. If that gate passes, merge according to repository policy, create the release tag, rebuild the committed-tree ZIP, and use the generated operator/evidence matrix and figures as the implementation basis for dissertation Chapter 4.

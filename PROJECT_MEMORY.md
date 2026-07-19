# Project Memory

- Date: 2026-07-19
- Branch: `feat/research-framework-completion`
- Implementation commit: `17e43f750c5100fd25148dd52aa49abbdc24f644`
- Release-check commits: `62b58f9`, `b0ae0c0`
- Tag candidate: `v1.0.0-rc1`

## Current focus

The primary product is the installable FuzzyXAI research framework and its reproducible evidence-first explanation pipeline. The generated DubnaXAI website is frozen and archived until the framework API, traceability, cross-model validation, and visual explanations are stable.

## Canonical public API

```python
fx = FuzzyXAI.wrap(model, adapter="auto", explain_plan=plan)
result = fx.explain(X)
result.plot()
result.export_json("explanation.json")
```

The canonical adapter contract is `ModelAdapter`. The canonical visualization namespace is `fuzzyxai.visualization`; `visual` and `viz` remain compatibility shims for one migration cycle.

## Implemented boundary

- callable, generic `predict_proba`, scikit-learn, and native-rule adapters;
- typed chapter 2-3 operator facade backed by the research core;
- machine-checkable operator manifest;
- data-quality evidence and explicit anomaly semantics;
- per-object training trajectories, forgetting events, and subgroup averaging diagnostics;
- native and surrogate rule extraction with provenance and limitations;
- measured rule ablation, class concepts, prototypes, coverage, similar cases, and counterfactual evidence;
- one `ExplanationGraph`, user/expert/audit narratives, provenance, and explanation-quality fields;
- one serializable `ExplanationViewModel` for Matplotlib and MATLAB/Octave;
- controlled object 85 protocol with measured global and rare-subgroup restoration effects.

## Claim boundary

- Do not claim universal model support. Native Torch, Keras, and ONNX adapters are not implemented.
- A surrogate rule is always labeled `surrogate`; it is not presented as model-internal knowledge.
- Similarity always states the compared representation and metric. It is not a probability or causal statement.
- Missing requested evidence produces `insufficient_evidence`; missing optional operator evidence produces `review` and explicit diagnostics.
- Medical examples are research demonstrations and have no clinical or production certification claim.
- MATLAB compatibility is limited to loading the canonical JSON and reproducing visual panels; the mathematical core remains Python.

## Repository policy

- The historical generated site is preserved by the archive branch `archive/site-prototype-cab4018` and excluded from source releases.
- Generated reports, caches, virtual environments, and local IDE state must not enter release ZIP files.
- Build a source release only from a committed tree with `python scripts/build_framework_release.py`.
- Create a release tag only after the public Python 3.11/3.12 and Octave GitHub Actions jobs are green.

## Validation

- local regression environment: `299 passed`;
- clean Python 3.11 checkout: `299 passed`, framework gate PASS, wheel/sdist PASS, wheel import PASS;
- clean Python 3.12 checkout: `299 passed`, framework gate PASS, wheel/sdist PASS, wheel import PASS;
- operator manifest: `30/30`, PASS;
- controlled object 85 protocol: PASS;
- MATLAB files included in the wheel: PASS;
- MATLAB/Octave execution: PASS in public GitHub Actions;
- public GitHub Actions run `29668215392`: Python 3.11 PASS, Python 3.12 PASS, Octave PASS.

## Next step

Use the implementation evidence to write dissertation chapter 4 as a formula-to-code-to-test-to-artifact argument. Resume the DubnaXAI ecosystem and chapter 5 only after the framework release boundary is stable.

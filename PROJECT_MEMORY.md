# Project Memory

- Date: 2026-07-19
- Branch: `feat/explanation-experience`
- Implementation commit: `17e43f750c5100fd25148dd52aa49abbdc24f644`
- Release-check commits: `62b58f9`, `b0ae0c0`
- Release tag: `v1.0.0-rc1`
- Explanation Experience base commit: `e84373881e602f6fb92ffe4874baa5bc9c21db33`
- v1.1 candidate implementation commit: `60b86ee`

## Current focus

The primary product is the installable FuzzyXAI research framework and its reproducible evidence-first explanation pipeline. The current milestone is `v1.1.0-rc1: Explanation Experience`: turn correct evidence operators into a coherent, inspectable story of how data became a decision. The generated DubnaXAI website remains frozen and archived.

## Canonical public API

```python
fx = FuzzyXAI.wrap(model, adapter="auto", explain_plan=plan)
result = fx.explain(X)
result.plot()
result.export_json("explanation.json")
```

The preferred v1.1 surface is `result.overview()`, `result.story()`, `result.inspect(...)`, `result.audit()`, and `result.visualize(view=..., backend=...)`.

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

## Explanation Experience candidate

- `ExplanationClaim` is the only source of user/expert/audit sentences;
- `ExplanationGraph` explicitly links evidence to claims, diagnostics, and action;
- E0-E5 disclose available, missing, native, and surrogate channels;
- `ExplanationVisualSpec` separates scientific results from renderer input;
- focused views cover story, percentile data profile, training trace, knowledge atlas, decision evidence, similar cases, counterfactuals, rule ablation, provenance, and audit;
- Matplotlib and Plotly consume the same VisualSpec; MATLAB consumes the same JSON;
- controlled golden explanations cover object 85, native-rule ANFIS, and a research-only medical similarity fixture;
- the comprehension protocol exists, but its status is `planned_not_run`.

## Claim boundary

- Do not claim universal model support. Native Torch, Keras, and ONNX adapters are not implemented.
- A surrogate rule is always labeled `surrogate`; it is not presented as model-internal knowledge.
- Similarity always states the compared representation and metric. It is not a probability or causal statement.
- Missing requested evidence produces `insufficient_evidence`; missing optional operator evidence produces `review` and explicit diagnostics.
- Medical examples are research demonstrations and have no clinical or production certification claim.
- MATLAB compatibility is limited to loading the canonical JSON and reproducing visual panels; the mathematical core remains Python.
- E0-E5 report evidence depth, not model quality or a universal interpretability score.
- Medical IoU and embedding similarity are distinct metrics and never a probability of the same diagnosis.

## Repository policy

- The historical generated site is preserved by the archive branch `archive/site-prototype-cab4018` and excluded from source releases.
- Generated reports, caches, virtual environments, and local IDE state must not enter release ZIP files.
- Build a source release only from a committed tree with `python scripts/build_framework_release.py`.
- Create a release tag only after the public Python 3.11/3.12 and Octave GitHub Actions jobs are green.

## Validation

- current local regression: `303 passed` on Python 3.14;
- Explanation Experience focused suite: `24 passed`;
- deterministic golden rebuild and SHA256 verification: PASS;
- `1.1.0rc1` wheel build and isolated import: PASS;
- local regression environment: `299 passed`;
- clean Python 3.11 checkout: `299 passed`, framework gate PASS, wheel/sdist PASS, wheel import PASS;
- clean Python 3.12 checkout: `299 passed`, framework gate PASS, wheel/sdist PASS, wheel import PASS;
- operator manifest: `30/30`, PASS;
- controlled object 85 protocol: PASS;
- MATLAB files included in the wheel: PASS;
- MATLAB/Octave execution: PASS in public GitHub Actions;
- public GitHub Actions runs `29668215392` and `29668320459`: Python 3.11 PASS, Python 3.12 PASS, Octave PASS.

The public runs above validate `v1.0.0-rc1`; public CI for commit `60b86ee` is still pending. Octave was not installed in the local v1.1 environment.

## Next step

Run full regression, clean-checkout wheel/Octave validation, public CI, and the documented comprehension study. Create the `v1.1.0-rc1` tag only after the public checks are green. Then use the claim/graph/VisualSpec evidence to write dissertation chapter 4. Keep DubnaXAI quarantined until this release boundary is stable.

# FuzzyXAI observer: instance-bound explanations and model-response evidence

## What this validates

The observer has two independent, opt-in gates, both operating before
budget allocation and automatic acceptance.

1. **Local identity/shape:** compares the explanation's object ID, model
   version, predicted target class, local scope, ordered feature axis,
   contribution width and finite numeric values, and a declared canonical
   source digest with the prediction/route contract.
2. **Linear additive response:** for the declared protocol
   linear_logit_single_feature_reference_v1, compares each contribution
   against the independently measured change in the same target-class
   logit after replacing one feature with a fixed reference. It also
   verifies additive reconstruction from the all-reference score.

The first gate cannot detect an attribution value permutation if the shape
is unchanged. The second can detect it for the supported additive
linear-logit protocol.

The model-response gate is **not valid for arbitrary SHAP, LIME, nonlinear
predictors, interactions, conditional explanations or causal claims**.
Those require separately validated verification protocols.

## SDK binding

Construct LocalExplanationEvidence, attach it to
ExplanationArtifact.local_evidence, and add ordered feature_names to
PredictionArtifact. Enable DeploymentContext(require_local_evidence=True).

For the optional additive model-response gate, collect
LocalEffectEvidence using collect_linear_reference_effects with a trusted
source-model target-logit callable (one original, one all-reference, and
one per-feature reference-perturbation call), then attach it
to RouteArtifacts.observed_local_effects. Enable
DeploymentContext(require_local_effect_verification=True).

Inconsistent evidence yields HardGuardStatus.BLOCKED. Missing required
evidence yields HardGuardStatus.REVIEW_REQUIRED. Legacy API defaults
remain compatible; strict gates must be **enabled deliberately**.
Provided evidence is still checked even if not required.

SHA256 is an identity declaration. This observer compares it with a
canonical digest declaration, but **does not hash the original source
bytes**. Trusted source adapters must independently verify those bytes.
A forged but internally consistent declaration can pass.

## Developmental experiment

    python -m pytest -q tests/practical_controller/test_local_evidence_identity.py tests/practical_controller/test_local_effects.py tests/practical_controller/test_observer_real_models.py tests/selective_observer/test_oof_fold_isolation.py
    python experiments/observer_local_evidence_benchmark.py

The experiment trains stratified-split logistic regressions on sklearn
Breast Cancer, Wine, and Digits. Each held-out sample has its target-class
logit decomposed into local contributions. An independent score-response
route reruns the model after replacing individual features. Controlled
faults cover object, class, axis, width, scope, missing target, digest,
and same-shape value permutations.

Use fault-family counts and model-specific counts. Corruptions derived
from a single held-out sample are correlated; do not treat each as an
independent data point.

Baselines are:
- declared object/model/source identity;
- local structural binding;
- local structural binding plus independent model responses.

This is **developmental fault injection**, not preregistered confirmatory
testing. It does not show unknown-fault detection, arbitrary-explainer
faithfulness, predictive accuracy or improved human decisions.

## Independent CV correction

The selective risk controller's grouped out-of-fold predictions now use
a StandardScaler fitted only on each fold's training groups. The final
development model still normalizes using all development data.

Complete CI and release smoke checks are required before merge.

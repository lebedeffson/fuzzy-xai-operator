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


## Stronger baselines and rehashed wrong explanations

The benchmark also measures a **structural plus logit-sum reconstruction**
baseline. A zero-sum transfer between two distinct feature contributions
preserves the overall target-logit reconstruction, and thus may pass this
otherwise sensible baseline. The full verifier checks the individual
model responses for each feature.

For every synthetic *wrong explainer output*, the benchmark recomputes a
consistent SHA256 over the **actual declared incorrect contributions**,
and updates the enclosing artifact identities. A separate digest-mismatch
case is retained. This explicitly models an explainer that emitted and
hashed a wrong but well-formed result; a byte-integrity checker cannot
detect that semantic error. It does **not** model corruption in transit
against a separately trusted immutable digest.

Zero-sum transfers are randomized after model fitting using a separate
fixed generator, with deltas 0.0001, 0.01 and 0.1. These are still
developer-defined mutation families, not natural failures or independent
external incidents. Report all dataset-by-family outcomes and explain that
samples within one fitted model are correlated.


## Optional input-content binding

Object identifiers may be reused for a different feature vector.
With DeploymentContext(require_input_binding=True), the guard requires
matching SHA256 hashes of the *ordered numeric feature input*: from the
prediction, local explanation, and source-model effect probes when present.
The hash_numeric_input helper uses deterministic encoding for finite
tabular numeric vectors. The new benchmark includes stale-input reuse
with the **same object ID, target, model version and feature axis**,
plus a separate missing-input-hash scenario.

A mismatched input hash blocks; a missing required hash asks for review.
A digest is a declaration and is not an authenticated signature. The
prediction and explainer adapters must compute their hashes from actual
data, not copy the hash out of each other's output. Non-numeric inputs
require another canonicalization protocol.


## Vectorized source-model probing

For a d-feature linear-additive target-logit model, the sequential
collector needs d+2 Python-level model invocations (the original input,
the all-reference input and one replacement per feature). The new
collect_linear_reference_effects_batched API sends the same d+2 input rows
to a vectorized source-model adapter in **one call**. It checks the output
length, finite scores and input-content hash and produces the exact same
typed LocalEffectEvidence.

The algorithm still evaluates O(d) score rows: batching reduces call
overhead, not mathematical complexity. Runtime speed depends on the
source model and hardware, so no general speedup is claimed without
independent measurements. The built-in sklearn benchmark uses this
batched collector and records both model call and score-row counts.


## Fitted-model identity and reproducibility metadata

The experiment now records full SHA256 fingerprints over actual fitted
LogisticRegression coefficients/intercepts, class axis, and StandardScaler
means/scales. These model digests are included in the model-version
declaration; the random seed alone is **not** the identity of a trained
model. The JSON also includes Python, NumPy and scikit-learn versions,
sample counts and feature dimensions. These fingerprints identify the
reported fitted state; they do not authenticate a remotely supplied model.

Additional seeded runs on the same three datasets test implementation
stability. They are exploratory and must not be described as independent
external validation.


## Nonadditive negative control

A direct interaction model f(x)=x0*x1 at x=(1,1) with reference (0,0)
has model-observed single-feature effects (1,1), while the total score
change is 1. The additive gate must reject this inconsistent
decomposition rather than certifying it. This test demonstrates detection
of one specified interaction, not general automatic recognition of every
nonlinear model. In particular, a locally zero or carefully balanced
interaction may remain invisible to the specified probes.


## Explainability background/reference identity

A legitimate explainer can use a different background reference from
the model's training mean. For example, SHAP's default Independent
masker may sample 100 background rows. Comparing correct SHAP values to
model probes taken at an **incorrect reference** yields a false
rejection, even when object, model and target class agree.

The optional require_reference_binding=True contract now compares the
background/reference SHA256 declared by the explainer with the SHA256
computed from the actual model-probe reference. Different references
block with LOCAL_REFERENCE:reference_sha256_mismatch; absent required
references request review. The canonical local record digest also
includes this background identity.

The actual native SHAP LinearExplainer regression uses the masker
background mean as its reference and tests genuine class-specific
linear-logit attributions. It also verifies that a consistently
rehashed but misdeclared reference is blocked. This is a **binary
logistic/linear-additive** validation, not evidence that arbitrary
SHAP/LIME configurations are faithful.


## Conditional linear-reference rebasing (experimental)

The public SDK `fuzzyxai.selective_observer.propose_linear_reference_rebase`
returns a **candidate**, never a certified or silently modified artifact.

For one fixed target-class linear-additive logit
`f_t(x)=b_t+sum_j w_j*x_j`, local contributions at reference
`r` are `phi_j(x;r)=w_j*(x_j-r_j)`. If the background changes from
`r0` to `r1`, the only justified algebraic transformation is

`phi_j(x;r1)=phi_j(x;r0)+w_j*(r0_j-r1_j)`.

The API requires actual input vector, old/new ordered reference vectors,
source attribution record, and **signed raw-space coefficients** from the
target-class model. Before returning a proposal, it verifies the input hash,
old reference hash, axis, finite values, and that the original attributions
match the raw-space linear contributions under `r0`. Contradictions yield
`incompatible`; missing required evidence yields `insufficient_evidence`.

A proposal is not automatically applied, hashed or permitted for action.
The caller must create a new explanation artifact with a **new canonical
source digest**, collect source-model responses against the **new** background,
and pass the usual strict identity + reference + response guards. Reusing
old source-model responses with a rebased explanation must be blocked.

Native SHAP LinearExplainer compatibility regression now includes the
target classes 0/1 on Breast Cancer, 0/2 on Wine, and 0/9 on Digits,
where independent-masker background sizes 50 and 100 produce distinct
attributions. This test demonstrates a mathematically exact reference
transformation for a fixed linear target logit, not general SHAP repair.
For nonlinear models, feature interactions, conditional expectations or
ambiguous baseline choices, this transformation is **not valid** and
should not be advertised or used.

Any measured speedup of batch inference or user-benefit improvement remains
outside the validated protocol.


## Exact empirical-interventional Shapley protocol (bounded nonlinear audit)

A separate opt-in protocol verifies instance-specific SHAP attributions
against independent calls to a source model **for one explicitly defined
empirical marginal replacement game**. It is not the earlier linear
single-feature-drop oracle and it does not apply to all SHAP variants.

For a fixed source-model scalar output f_t, a numeric input x with d
named features and an empirical background B of m finite feature vectors,
define

    v_t(S; x, B) = (1/m) * sum_{b in B} f_t(x_S, b_{not S}).

For each feature j, the reference contribution is the classical Shapley
value

    phi_j = sum_{S subset [d]\{j}} |S|! (d-|S|-1)! / d!
                * (v_t(S union {j}) - v_t(S)).

This is an existing mathematical identity, not a newly invented
attribution algorithm. It covers **nonlinear interactions** relative to
the specified game; individual one-feature response differences are
generally NOT its Shapley contributions. For example, f(x)=x0*x1 at
(1,1) with background {(0,0)} has Shapley=(0.5,0.5), even though the two
single-feature drops are (1,1).

### Integration and fail-closed behavior

Call fuzzyxai.selective_observer.collect_exact_interventional_shapley
with the source object, model version, one fixed scalar target output,
ordered numeric input axis, the full background rows, and a trusted
batch target_scores callable. It returns typed ExactShapleyEvidence
containing per-feature contributions, v(empty), v(full), observed input
SHA256, full-background SHA256, number of background rows and number of
source-model scores.

The corresponding local explanation must carry the same input_sha256,
background_sha256, object/model/target identity, axis and contributions;
its source digest must be freshly computed by the trusted adapter.
Attach the oracle to RouteArtifacts.observed_exact_shapley and enable
DeploymentContext(require_exact_shapley_verification=True).

- Any measured identity/background/value conflict yields
  HardGuardStatus.BLOCKED and precise EXACT_SHAPLEY:* reason codes.
- Absent required evidence or unsupported explanation game yields
  HardGuardStatus.REVIEW_REQUIRED; it is not silently certified.
- The existing linear-additive gate remains a distinct optional
  protocol and must not be substituted for a nonlinear Shapley game.
- The independent source adapter must authenticate model bytes and the
  selected score channel. The observer cannot prove that a declared
  callback ran the claimed source model merely because two hashes agree.

The exhaustive oracle evaluates 2**d * m score rows with a single
vectorized adapter invocation. Production limits: 1 <= d <= 8, at
most 8192 scored rows. This does not scale to unrestricted feature
counts, conditional distributions, causal attributions, stochastic
models or other attribution games.

Two background samples with the same feature-wise mean can give
different Shapley contributions to interacting models: the full
empirical background is therefore hashed, not only its mean.

### Developmental nonlinear comparison

Run: python -m experiments.observer_nonlinear_shap_benchmark

The experiment fits DecisionTree/RandomForest classifiers on Breast
Cancer, Wine and Digits with four selected raw numeric features and a
fixed eight-row background per dataset. It checks real SHAP
ExactExplainer contributions against independently enumerated
source-model coalition values.

Actual GitHub Actions artifacts from run 43111380695 (Python 3.11 and
3.12) contain 36 held-out objects, 216 controlled variants, 36/36 clean
certifications, 72/72 rehashed zero-sum corruption blocks, 36/36
wrong-background blocks, 36/36 stale-input blocks, and 36/36
missing-background review states. Structural-plus-score-sum admits all
72 zero-sum corruptions and all 36 wrong-background declarations.
Maximum source-vs-native numerical discrepancy: ~1.11e-16. The same
outcome counts were observed in both Python versions.

**Scientific limit:** the same characteristic game is evaluated by
native SHAP and our separate enumerator. Deliberately injected faults
remain correlated within each object; these records are not independent
natural incidents. Predictive accuracy, general explainer fidelity,
calibration, user comprehension and causal validity are unmeasured.
The next research step is independently specified held-out faults,
scoring-channel semantics and multiple background sampling strategies.


### Empirical background identity is a multiset

A legitimate reordering of the same empirical background rows does not
change the interventional Shapley game. Hashing an *ordered* background
matrix would therefore create false provenance blocks. The experimental
exact-Shapley background hash now canonically sorts finite numeric rows
and normalizes signed zero, while retaining duplicates as empirical
weights. Two permutations of the same multiset hash equally; different
multiplicities and different joint distributions do not.

This intentionally uses the explicit
empirical-background-multiset-v2 hash schema. Previously issued
ordered-background hashes should be regenerated and reverified before
reuse. No recorded locked/scored artifact is silently rewritten.

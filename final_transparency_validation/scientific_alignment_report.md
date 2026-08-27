# Scientific alignment report — P19

## Closed substitutions

P19 removes the P15-P18 shortcuts that were structurally convenient but
scientifically different from the dissertation operators:

- RF global feature importance is not a local reason.
- Gamma is not claim agreement or direct comparison without transformation.
- Delta is not top-k presentation loss or reconstruction fidelity.
- Predictive uncertainty is not surrogate fidelity.
- `U_model` is not the aggregate `u_M`.
- `I_pre` is not computed from an arbitrary first local object.
- A partial normalized score is not the five-component rho.

## Implemented operator route

The canonical public runtime performs:

`prediction -> E_model -> T_ij(E_model) -> E_target -> Gamma`, followed by
the independent `U_model/U_rules/U_trace -> u_M` aggregation, a typed
uncertainty representation and real reduction, composition into `E_pre`,
`I_pre(E_pre)`, strict five-term rho, critical policy, and action.

The model-side input is a typed, domain-neutral `SystemSourceEvidence`.
Adapters/providers translate native votes or class probabilities into this
contract; `system_semantics.py` consumes source interface terms, values,
activations, uncertainty inputs, trace and refs without knowing RF, medical
class names, or a particular dataset. Both RF and LogisticRegression public
routes exercise the same operator implementation.

`AlignmentTransform` is executable and serializable. Required alignment fails
closed without its execution. Gamma stores the component vector and policy
weights; `d_L` is disclosed as diagnostic-only because the validated beta
aggregate contains representation, rules, activations, uncertainty, and trace.

The system representation is `F_int`. Reduction records source interval,
midpoint Pi, reduced F0, inverse diagonal embedding, distance terms, and Delta.
The accept/conflict objects have degenerate intervals, so Delta=0 is a real
lossless result. A separate held-out RF object has votes 107/93, source interval
`[0.0365390, 1]`, and measured `Delta=0.4817305`; this demonstrates actual
information loss without injecting a probability or Delta.

Uncertainty evidence records method, inputs, formula, sources and status for
all three channels. RF `U_model` is the standard deviation of binary tree
votes (expected range `[0, 0.5]`), not variance. `U_trace` combines required
field checks with an explicitly sourced, externally verified trace status; it
does not claim independent verification when the caller supplied that status.
ExplainPlan eta weights are preserved with term contributions. `I_pre` records
its H/C/O/K/U components and weights.

The uncertainty representation is a declared validation-preset heuristic:
`[p_risk-scale*U_model, p_risk+scale*U_model]`, clipped to `[0,1]`. The policy
is owned by ExplainPlan, serialized with `calibrated=false`, and is not called a
calibrated prediction interval.

The risk contract is exactly:

`rho = w_p*rho_p + w_u*u_M + w_I*(1-I_pre) + w_Delta*Delta + w_R*chi_R`.

All five values and weights remain present in complete system cases. A
critical rupture forces block but does not erase numeric rho. Non-system cases
with unavailable non-zero-weight components export `rho=None` and may expose a
separately named partial score.

An explicitly zero-weight missing component is `not_applicable` and does not
make rho incomplete; no weight is redistributed. The four ExplainPlan
thresholds are all executable: accept, lower_confidence, configurable
request_more_data/review, configurable defer_to_human/review, then block at
theta_4. Candidate and final reasons are stored separately, and critical
override remains an independent final block path.

## Runtime and provenance

`FuzzyXAI.wrap(...).explain_one(...)` creates typed `SystemEvidence` once.
Reader report, audit, serialization, inspect and graph use the same stored
evidence. System generators perform no post-result system-operator
calculations. The image generator may derive caller-defined connected regions
from the public attribution tensor before registering those masks; it does not
recompute Gamma, uncertainty, Delta, I_pre, rho, action, or IG completeness.

ExplainPlan is the canonical policy owner. A system route fails closed when
alignment is inapplicable, its registered transform differs from the factual
ObservationContext handle, the requested uncertainty signal is unavailable,
or a declared reduction method is unsupported. ObservationContext retains
factual trace evidence and backward-compatible registration handles only.

The registered external-tabular route is explicitly namespaced as a legacy
compatibility route: top-k omission is `presentation_omission_loss`, its max
score is `legacy_route_score`, and it exports none of the canonical names
Gamma, Delta, or rho. Public `fuzzyxai.observe_risk` accepts only the strict
five-component P19 schema and never silently renormalizes weights.

The focused system graph uses actual runtime edges: the three uncertainty
channels independently converge on `u_M`; alignment comparison creates Gamma;
the five numeric rho inputs are explicit. Numerical rho yields a candidate
action, while `chi_R` independently yields `critical_override`; policy
resolution combines those facts into the final action. No dataset-training
edge is fabricated when it was not observed.

## Additional closure evidence

- Same-run training observer: actual partial-fit epochs, final-model
  fingerprint, first-learned/forgetting/stability and typed loss status.
- Image: full 28x28 Integrated Gradients tensor and a six-point convergence
  calculation in the same logit space as the attributions. The target class is
  fixed once from the source output and remains fixed along the path.
- Packaging: installed wheel discovers `operators_manifest.yaml` under its
  installed data path without a source checkout.
- Packaging excludes six disconnected research/Q1/closure namespaces after a
  static import-graph audit. `fuzzyxai.experiments` remains because seven
  defended manifest operators import installed callables from it. NiceGUI is
  an optional `ui` extra rather than a base runtime dependency.

## Known limitations

- The final 512-step Fashion-MNIST completeness relative residual is
  0.0001606. It is measured, not declared exact; the full convergence sequence
  is retained.
- Accept/conflict remain intentionally simple lossless reductions; positive
  Delta is demonstrated by the separate non-degenerate reduction case.
- Conflict is controlled trace fault injection, not a naturally discovered
  model conflict.
- RF disagreement is the standard deviation of binary indicators relative to
  the registered risk class, hence invariant to numeric or string labels.
  `rho_p` uses model probability when available; hard votes remain separate
  uncertainty evidence. `chi_R` and `chi_R_critical` are persisted separately.
- Optional dependencies absent from the regression environment remain skipped
  and are listed in the regression log.

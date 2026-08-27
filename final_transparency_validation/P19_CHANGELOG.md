# P19 changelog — System Operator Closure

- Removed RandomForest global `feature_importances_` from local-reason semantics;
  RF evidence is per-tree predictions, vote proportions, and disagreement.
- Added executable, serializable `AlignmentTransform`; Gamma is now
  `d_E(T_ij(E_model), E_target)` and cannot be certified without execution.
- Added separate typed `U_model`, `U_rules`, `U_trace` evidence and
  ExplainPlan-controlled eta aggregation into `u_M`.
- Added real `F_int -> Pi -> F0 -> iota -> D_F -> Delta` reduction evidence,
  first-class `E_pre`, auditable `I_pre(E_pre)`, and strict five-term rho.
- Critical rupture now overrides action without deleting the numeric rho.
- Split risk decision provenance into numeric candidate action and independent
  critical override, then explicit policy resolution into the final action.
- Wired the complete system route into canonical
  `FuzzyXAI.wrap(...).explain_one(...)`; exporters only project the returned
  `ModelExplanationResult`.
- Added runtime-created system graph nodes and a four-row focused action
  renderer. `U_model`, `U_rules`, and `U_trace` independently converge on
  `u_M`; Gamma enters the recorded risk ancestry.
- Added public accept and controlled fault-injection conflict cases.
- Added a public non-degenerate reduction case from a real held-out object:
  107/93 tree votes, `U_model>0`, nonzero interval width, and `Delta>0`.
- Added a same-run `SGDClassifier.partial_fit` training case with run ID,
  final-model fingerprint, measured epochs, forgetting reason, and loss state.
- Fixed Integrated Gradients target drift along the interpolation path, added
  configurable trapezoidal interval counts, and retained 16..512 convergence
  measurements for the 28x28 Fashion-MNIST case.
- Fixed triangular shoulder endpoints and wheel-only manifest discovery.
- Removed the legacy silently renormalized local score from the name rho;
  incomplete local routes expose `rho=None` and `partial_risk_score`.
- Enforced ExplainPlan ownership of alignment, uncertainty, representation,
  reduction, and system membership policies; conflicting ObservationContext
  handles now fail closed.
- Split channel disclosure into `required_missing`, `optional_missing`,
  `not_applicable`, and `available`; optional training/counterfactual channels
  no longer contradict a complete system accept result.
- Corrected RF capability projection: native ensemble votes and disagreement
  do not imply per-feature `local_contributions`.
- Namespaced the registered external tabular max/omission route as legacy;
  it no longer exports top-k omission as Delta or a max score as rho.
- Made public `fuzzyxai.observe_risk` enforce the strict five-component P19
  contract and removed implicit risk-weight normalization.
- Disclosed the RF uncertainty signal as binary-vote standard deviation, the
  interval construction as a non-calibrated ExplainPlan heuristic, and trace
  completeness as externally verified factual evidence.
- Added IG completeness to the public reader report without reporter-side
  recomputation.
- Replaced RF/BCW construction inside the operator core with typed
  `SystemSourceEvidence`; RF votes and non-RF class probabilities now enter the
  same public runtime through adapter-side providers.
- Activated all four ExplainPlan rho thresholds, persisted candidate/final
  action reasons, and allowed zero-weight missing components without hidden
  renormalization.
- Removed stale runtime-values and architecture-matrix artifacts, unified the
  public interval notation as `F_int`, and corrected rho provenance to its five
  actual inputs.
- Excluded six disconnected dissertation namespaces from the wheel and moved
  NiceGUI to the optional `ui` extra. Retained `fuzzyxai.experiments` because
  seven defended manifest operators resolve their installed callables there.
- Made RF vote dispersion label-invariant by computing standard deviation of
  risk-class vote indicators; separated model probability from hard-vote
  proportion; separated `chi_R` from `chi_R_critical`; and made the public
  missing-required projection ignore absent zero-weight risk components.

Final validation: 1261 passed, 11 skipped, 643 warnings in 288.12 s; scoped
mypy/ruff and compileall passed; the installed wheel passed manifest, local,
RF-system, non-RF-system, report, audit, graph, inspect and serialization smoke
checks from a temporary directory outside the source checkout.

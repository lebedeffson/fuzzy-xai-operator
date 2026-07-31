# Limitations

The evaluation uses eighteen fixed registered mutations. It does not establish detection of arbitrary ML errors, human-time reduction, user utility, or transfer to natural software incidents. MLflow and FuzzyXAI have different roles.

The scenario-paired bootstrap intervals for the O-versus-B2 diagnostic
differences excluded zero, but Holm-adjusted McNemar p-values did not. The
small, deterministic scenario registry is suitable for component ablation and
acceptance testing, not population-level inference. No claim is made that the
observed proportions transfer to unregistered mutations or natural incidents.

B2 intentionally includes every registered local schema, finite-value,
feature-count, hash, metric-range, convergence, required-field and SHAP
consistency check. Its limitation is only the absence of cross-stage relations,
RouteGraph diagnostic cuts and full-route recertification.

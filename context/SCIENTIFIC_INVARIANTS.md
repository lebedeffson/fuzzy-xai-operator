# Scientific invariants

## Rejected shortcuts

- DO NOT use `feature_importances_ * zscore` as RF local contribution.
- DO NOT use direct d_E(E_i, E_j) where T_ij is required.
- DO NOT use claim agreement as Gamma.
- DO NOT use reconstruction error or top-k omission as dissertation Delta.
- DO NOT use surrogate fidelity as U_model or u_M.
- DO NOT substitute U_model for aggregated u_M.
- DO NOT compute I_pre from an arbitrary local object.
- DO NOT silently drop risk components and still call the result rho.
- DO NOT fabricate graph nodes, evidence, metrics, or provenance.

Gamma requires executed E_i -> T_ij(E_i) -> E_j comparison. Delta requires a
real uncertainty-representation reduction and inverse embedding. U_model,
U_rules, and U_trace are distinct; ExplainPlan defines u_M aggregation.
Critical rupture is never numerically compensable.

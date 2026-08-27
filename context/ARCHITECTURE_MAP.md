# Architecture map

Canonical public route: `FuzzyXAI.wrap(...).explain(...)` or `.explain_one(...)`.

For P19 system semantics:

`prediction -> E_model -> T_ij -> aligned E_model -> E_target -> Gamma ->
uncertainty profile -> uncertainty representation -> reduction -> E_pre ->
I_pre -> rho -> action`.

Evidence collectors observe; `core` owns mathematics; `operators` is a typed
facade; visualization renders computed values only. Add a graph node only when
the backing evidence exists.

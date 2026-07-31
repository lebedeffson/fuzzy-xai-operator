# ML Vertical v1 acceptance criteria

The status is `FUZZYXAI_ML_VERTICAL_V1_IMPLEMENTED` only when every condition below passes. Otherwise it is `FUZZYXAI_ML_VERTICAL_V1_ACCEPTANCE_FAILED`.

1. The Docker Compose configuration exposes API, defense UI, MLflow and persistent artifacts.
2. A real fitted sklearn `LogisticRegression` produces the prediction.
3. Real `shap.LinearExplainer` produces the local explanation and satisfies its output consistency tolerance.
4. The public canonical `ExplainableObject` contains evidence, claims, provenance, fuzzy state and a canonical hash without manual Gold insertion.
5. S1-S10 exercise `F0`, `F_int`, `NAS`, and `F_ML` and all five observer actions where registered.
6. Representation selection follows the frozen uncertainty profile and plan.
7. Reduction loss is measured and changes the observer action when above `0.25`.
8. Every user claim has a graph path to versioned Evidence.
9. Every S1-S10 expected action/issue/repair/determinism condition passes through the public API.
10. Critical-issue false certification is exactly zero.
11. Repeated S10 requests have the same canonical structured hash.
12. User text is deterministic and references only registered claim codes with evidence.
13. User, engineer and auditor views reference one explainable-object hash.
14. MLflow records all registered tags, metrics and nine JSON artifacts without network access.
15. All pre-existing Chapter 4 result/report/protocol files remain byte-identical.
16. Focused tests, full regression, Ruff changed-scope, compileall, claim lint, operator manifest and clean-source tests pass.

Machine runtime is descriptive and must not be described as user or specialist time. The Breast Cancer dataset is a software demonstration and does not support clinical claims.

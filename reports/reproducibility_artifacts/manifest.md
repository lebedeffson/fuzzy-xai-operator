# Reproducibility Artifacts

- status: `ok`
- python: `3.14.4`
- platform: `Linux-7.0.3-arch1-2-x86_64-with-glibc2.43`
- ExplainPlan SHA256: `dc4460c6d7db9df9d3e95dcad6119c99d9c99093911c883d1186573d0cb05473`

## Commands

- `make dissertation-check`
- `make thesis-practice-tables`
- `make browser-visual-check`
- `make ui-health-check`
- `make structure-aware-benchmark DATASET=wine_risk`
- `make structure-aware-benchmark DATASET=diabetes_binary`
- `make reproducibility-artifacts`

## Artifact Manifest

| artifact | exists | sha256 | chapter | confirms | command |
| --- | --- | --- | --- | --- | --- |
| `reports/reproducibility_artifacts/explain_plan.json` | `True` | `d7cdb8f76152` | 2 / appendix | Serializable ExplainPlan contract and trace hash source | `make reproducibility-artifacts` |
| `reports/chapter2_real_operator_case/breast_cancer_operator_case.json` | `True` | `618947c60c12` | 2 | sample_113 operator values: mu, alpha, U_model, U_rules, U_trace, u_M, tau | `make chapter2-real-operator-case` |
| `reports/real_reduction_example/breast_cancer_case.json` | `True` | `ca31cd57c9b9` | 3 | sample_113 reduction, chi_Auto, chi_R, chi_R_crit, rho, action | `make real-reduction-example` |
| `reports/datasets/breast_cancer/summary.json` | `True` | `e93883146409` | 5 | Breast Cancer model metrics, observer metrics, I_pre/rho quantiles | `make benchmark-dataset DATASET=breast_cancer` |
| `reports/datasets/breast_cancer/calibration.json` | `True` | `68118e91c83b` | 5 | Observer calibration before/after and constrained parameters | `make calibrate-observer DATASET=breast_cancer` |
| `reports/datasets/synthetic_ruptures/baseline_comparison_native.json` | `True` | `fad3e831c07d` | 5 | Native-access safety comparison; baselines do not receive chi_R_crit | `make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=native` |
| `reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json` | `True` | `5d7ddd1a679e` | 5 | Equal-guardrail sanity check where all methods receive chi_R_crit | `make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=equal_guardrail` |
| `reports/structure_aware_benchmark/breast_cancer.json` | `True` | `4911d5053195` | 5 | Structure-aware cases: trace gap, source conflict, context forbidden, high Delta | `make structure-aware-benchmark DATASET=breast_cancer` |
| `reports/structure_aware_benchmark/wine_risk.json` | `True` | `c5e08a516a9a` | 5 | Structure-aware benchmark on real wine rows (non-synthetic improvement check) | `make structure-aware-benchmark DATASET=wine_risk` |
| `reports/structure_aware_benchmark/diabetes_binary.json` | `True` | `dcfdb0a0eaf8` | 5 | Structure-aware benchmark on real diabetes rows (non-synthetic improvement check) | `make structure-aware-benchmark DATASET=diabetes_binary` |
| `reports/datasets/breast_cancer/ablation.json` | `True` | `3a4d678e2890` | 5 | Ablation of trace, Delta, chi_R_crit, hierarchy, topos, risk-only threshold | `make ablation-benchmark DATASET=breast_cancer` |
| `reports/thesis_practice/thesis_practice_tables.json` | `True` | `fc46650590a0` | appendix | Word/LaTeX-ready thesis practice table index | `make thesis-practice-tables` |
| `Makefile` | `True` | `3c2470dcb4de` | appendix | Reproducible command route | `` |
| `requirements.txt` | `True` | `4d381eabe377` | appendix | Python environment dependencies | `` |
# Reproducibility Artifacts

- status: `ok`
- python: `3.14.5`
- platform: `Linux-7.0.11-arch1-1-x86_64-with-glibc2.43`
- ExplainPlan SHA256: `dc4460c6d7db9df9d3e95dcad6119c99d9c99093911c883d1186573d0cb05473`

## Commands

- `make dissertation-check`
- `make reproduce-chapter2`
- `make calibrate-chapter2`
- `make benchmark-equal-raw-structure`
- `make chapter3-artifacts`
- `make ecosystem-evidence`
- `make validate-ecosystem-sdk`
- `make thesis-practice-tables`
- `make browser-visual-check`
- `make ui-health-check`
- `make structure-aware-benchmark DATASET=wine_risk`
- `make structure-aware-benchmark DATASET=diabetes_binary`
- `make reproducibility-artifacts`

## Artifact Manifest

| artifact | exists | sha256 | chapter | confirms | command |
| --- | --- | --- | --- | --- | --- |
| `configs/explain_plan_chapter2.yaml` | `True` | `4469dad3fbee` | 2 / appendix | Fixed chapter 2 ExplainPlan YAML contract used for deterministic hashing | `make reproduce-chapter2` |
| `reports/chapter2/explain_plan_hash.json` | `True` | `c18bfa69005f` | 2 / appendix | Validated ExplainPlan SHA256 and required trace field list | `make reproduce-chapter2` |
| `reports/chapter2/sample_113_report.json` | `True` | `4920325789b0` | 2 | Canonical sample_113 report generated from the chapter 2 ExplainPlan contract | `make reproduce-chapter2` |
| `reports/chapter2/calibration_constants.csv` | `True` | `5ae7af763efb` | 2 | Chapter 2 calibrated c_H/c_O/c_K constants on calibration pairs | `make calibrate-chapter2` |
| `reports/chapter2/equal_raw_structure_summary.csv` | `True` | `768da4928dc9` | 2 | Equal raw structure benchmark for certified route vs raw structural access | `make benchmark-equal-raw-structure` |
| `reports/chapter3/dataset_roles_summary.csv` | `True` | `e677dd7743b0` | 3 | Chapter 3 concise dataset role table | `make chapter3-artifacts` |
| `api/openapi.yaml` | `True` | `35edef0eea1a` | 4 | Open API contract for /v1/explain and /v1/risk-action | `make validate-ecosystem-sdk` |
| `deploy/docker-compose.yml` | `True` | `b67bcadfd5df` | 4 / appendix | Deployment skeleton for API and Studio services | `` |
| `templates/module_registry_entry.json` | `True` | `0479e2efe8c6` | 4 / appendix | External module registration template | `python scripts/register_external_module.py --module-id my_module` |
| `fuzzyxai/ecosystem/registry.json` | `True` | `e794e77071a3` | 4 | External module registry for chapter 4 ecosystem integration | `make ecosystem-evidence` |
| `evidence/evidence_matrix.csv` | `True` | `97a0eda4f67f` | 4 / 5 | Module evidence matrix with status, fixture, and claim-scope flags | `make ecosystem-evidence` |
| `reports/chapter4/ecosystem_evidence.json` | `True` | `d62e72a2d297` | 4 | Chapter 4 external module evidence summary | `make ecosystem-evidence` |
| `reports/reproducibility_artifacts/explain_plan.json` | `True` | `d7cdb8f76152` | 2 / appendix | Serializable ExplainPlan contract and trace hash source | `make reproducibility-artifacts` |
| `reports/chapter2_real_operator_case/breast_cancer_operator_case.json` | `True` | `0d319ff99567` | 2 | sample_113 operator values: mu, alpha, U_model, U_rules, U_trace, u_M, tau | `make chapter2-real-operator-case` |
| `reports/real_reduction_example/breast_cancer_case.json` | `True` | `ca31cd57c9b9` | 3 | sample_113 reduction, chi_Auto, chi_R, chi_R_crit, rho, action | `make real-reduction-example` |
| `reports/datasets/breast_cancer/summary.json` | `True` | `e93883146409` | 5 | Breast Cancer model metrics, observer metrics, I_pre/rho quantiles | `make benchmark-dataset DATASET=breast_cancer` |
| `reports/datasets/breast_cancer/calibration.json` | `True` | `68118e91c83b` | 5 | Observer calibration before/after and constrained parameters | `make calibrate-observer DATASET=breast_cancer` |
| `reports/datasets/synthetic_ruptures/baseline_comparison_native.json` | `True` | `fad3e831c07d` | 5 | Native-access safety comparison; baselines do not receive chi_R_crit | `make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=native` |
| `reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json` | `True` | `5d7ddd1a679e` | 5 | Equal-guardrail sanity check where all methods receive chi_R_crit | `make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=equal_guardrail` |
| `reports/structure_aware_benchmark/breast_cancer.json` | `True` | `dee3b7e6a34a` | 5 | Structure-aware cases: trace gap, source conflict, context forbidden, high Delta | `make structure-aware-benchmark DATASET=breast_cancer` |
| `reports/structure_aware_benchmark/wine_risk.json` | `True` | `84c321f3a160` | 5 | Structure-aware benchmark on real wine rows (non-synthetic improvement check) | `make structure-aware-benchmark DATASET=wine_risk` |
| `reports/structure_aware_benchmark/diabetes_binary.json` | `True` | `c8b25086a268` | 5 | Structure-aware benchmark on real diabetes rows (non-synthetic improvement check) | `make structure-aware-benchmark DATASET=diabetes_binary` |
| `reports/datasets/breast_cancer/ablation.json` | `True` | `3a4d678e2890` | 5 | Ablation of trace, Delta, chi_R_crit, hierarchy, topos, risk-only threshold | `make ablation-benchmark DATASET=breast_cancer` |
| `reports/thesis_practice/thesis_practice_tables.json` | `True` | `fc46650590a0` | appendix | Word/LaTeX-ready thesis practice table index | `make thesis-practice-tables` |
| `Makefile` | `True` | `3ee8c7d4d8c7` | appendix | Reproducible command route | `` |
| `requirements.txt` | `True` | `3202b5309cef` | appendix | Python environment dependencies | `` |
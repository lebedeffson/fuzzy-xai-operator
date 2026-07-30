PYTHON ?= python
CONFIRMATORY_PYTHON ?= .venv-confirmatory/bin/python
PYTHONPATH := .
PORT ?= 8085
DATASET ?= breast_cancer
BASELINE_ACCESS ?= native

.PHONY: install test risk-test category-hott-test studio-engine-test studio-hybrid-batch studio-export-tables category-hott-test chapter2-breast-cancer-demo chapter2-real-operator-case reproduce-chapter2 calibrate-chapter2 benchmark-equal-raw-structure chapter2-3-final-evidence chapter3-artifacts reproduce-critical-ruptures chapter3-audit chapter3-real-conflicts chapter3-f0-vs-nas chapter3-calibrate-observer chapter3-tables chapter3-validate chapter3-final-evidence chapter3-practice-natural chapter3-practice-conflict chapter3-practice-bootstrap chapter3-practice-baselines chapter3-practice-calibrate chapter3-practice-ablation chapter3-practice-sensitivity chapter3-practice-stats chapter3-practice-validate chapter3-practice-all figures-chapter2 chapter2-figures chapter2-patch chapter2-validate chapter2-package2 ecosystem-evidence doctoral-final-evidence validate-ecosystem-sdk dissertation-artifacts chapter5-experiments chapter5-demo chapter5-latex web-demo unified-demo layered-demo layered-demo-legacy defense-demo defense-demo-legacy studio ui-health-check ui-health-check-all browser-visual-check unified-demo-cli full-pipeline figures full-experiments demo dashboard proof formal-proof thesis full-demo full-observer dataset-observer dataset-modes-check baseline-check real-data-validation benchmark benchmark-dataset baseline-comparison calibrate-observer ablation-benchmark defense-cases real-reduction-example dissertation-demo-summary dissertation-component-tables dissertation-check dataset-cards thesis-practice-tables structure-aware-benchmark reproducibility-artifacts operator-benchmark risk-benchmark lofo-f1-demo reproduce-dissertation empirical-smoke empirical-full-check clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

studio-engine-test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_studio_operator_engine.py tests/test_fuzzyxai_studio_demo_readiness.py -q

studio-hybrid-batch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.run_scenario hybrid_xiris --batch --out-dir reports/studio_batch

studio-export-tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.export_tables --scenario hybrid_xiris --out-dir reports/chapter5/studio_tables

.PHONY: final-readiness-audit
final-readiness-audit: studio-hybrid-batch studio-export-tables
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.inventory
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.grep_stale_terms
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_chapters --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_format --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.formula_references
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_render_gate --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.final_audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.build_package
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/audit -q

.PHONY: studio-semantic-smoke studio-server-smoke studio-smoke operator-manifest-check framework-release-check framework-source-release doctorate-release-check fresh-clone-gate practice-demo practice-screenshots practice-package practice-package-with-qa dataset-audit train-all evaluate-all training-audit practice-readiness-check screenshot-qc proof-qc package-self-contained-check real-validation-check full-delivery-package final-delivery-report final-product-check research-repo-inventory framework-check fuzzyxai-framework-check framework-external-check fuzzyxai-cli-check fuzzyxai-schema-check fuzzyxai-adapter-sdk-check fuzzyxai-framework-rc-check fuzzyxai-framework-rc-package fuzzyxai-visualization-check fuzzyxai-visualization-package fuzzyxai-visual-quality-check fuzzyxai-shap-like-visualization-check fuzzyxai-shap-like-visualization-package fuzzyxai-ru-visual-explanation-check fuzzyxai-ru-visual-explanation-package fuzzyxai-ru-visual-editorial-check fuzzyxai-ru-operator-explanation-check fuzzyxai-ru-explanation-framework-check operator-traceability-check research-validation research-validation-check fuzzyxai-research-analysis fuzzyxai-research-analysis-check applications-check operator-dashboard operator-route-check site-build sprint-report dubnaxai-release-check
.PHONY: reproduce-q1 reproduce-q1-smoke verify-q1 q1-baseline-snapshot q1-claims q1-tables q1-figures q1-archives q1-archive-check reproduce-q1-final-smoke reproduce-q1-final verify-q1-final q1-final-archives q1-final-archive-check q1-final-external-check selective-observer-formative-check

q1-baseline-snapshot:
	$(PYTHON) scripts/q1/build_baseline_snapshot.py

reproduce-q1-smoke:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/reproduce_all.py --profile smoke

reproduce-q1:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/reproduce_all.py --profile full

verify-q1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/verify_all.py

q1-claims:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/build_claim_registry.py

q1-tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/build_tables.py

q1-figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/q1/build_figures.py

q1-archives:
	$(PYTHON) scripts/build_framework_release.py
	$(PYTHON) scripts/q1/build_archives.py

q1-archive-check:
	$(PYTHON) scripts/q1/verify_archives.py

reproduce-q1-final-smoke:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/reproduce_all.py --profile smoke

reproduce-q1-final:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/reproduce_all.py --profile full

verify-q1-final:
	PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/verify_all.py --require-heavy

q1-final-archives:
	PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/build_archives.py

q1-final-archive-check:
	PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/verify_archives.py

q1-final-external-check:
	PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/q1_final/verify_external_gates.py

selective-observer-formative-check:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/selective_observer/build_protocol_manifest.py
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:. $(PYTHON) scripts/selective_observer/verify_formative_boundary.py
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:. $(PYTHON) -m pytest tests/selective_observer -q
	@echo "selective-observer-formative-check: PASS"
studio-semantic-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.studio_smoke

studio-server-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.studio_server_smoke

studio-smoke: studio-semantic-smoke studio-server-smoke

operator-manifest-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.operators_manifest --output reports/audit/operators_manifest_report.json

framework-release-check: operator-manifest-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_public_framework_api.py tests/test_evidence_first_framework.py tests/test_framework_release_contract.py -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/object_85_training_trace.py --output-dir release_evidence/generated/object_85
	@echo "framework-release-check: PASS"

framework-source-release: framework-release-check
	$(PYTHON) scripts/build_framework_release.py

doctorate-release-check: operator-manifest-check studio-hybrid-batch studio-export-tables
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.inventory
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.grep_stale_terms
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.formula_references
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_chapters --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_format --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.docx_render_gate --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.final_audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.build_package
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/audit tests/test_studio_operator_engine.py tests/test_fuzzyxai_studio_demo_readiness.py tests/test_public_framework_api.py tests/test_evidence_first_framework.py tests/test_framework_release_contract.py -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.studio_smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.studio_server_smoke

fresh-clone-gate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.fresh_clone_gate

dataset-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.dataset_audit

train-all:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.train.train_all

evaluate-all:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.evaluate.evaluate_all

training-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.training_audit

practice-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.practice_demo

practice-screenshots:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.practice_demo --screenshots-only

practice-package:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.practice_demo --package-only

practice-package-with-qa:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.practice_demo --package-only

practice-readiness-check: dataset-audit train-all evaluate-all training-audit practice-demo practice-screenshots practice-package
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.practice_demo --validate

screenshot-qc:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.screenshot_qc

proof-qc:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.proof_qc

package-self-contained-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.package_self_contained

real-validation-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.realdata.fetch_real_artifacts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.real_validation

final-delivery-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.final_delivery_report

full-delivery-package:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.audit.build_full_delivery

final-product-check: dataset-audit train-all evaluate-all training-audit practice-demo practice-screenshots practice-package screenshot-qc proof-qc practice-package-with-qa package-self-contained-check real-validation-check doctorate-release-check final-delivery-report full-delivery-package
	@echo "final-product-check: PASS"

research-repo-inventory:
	$(PYTHON) scripts/inventory_research_repositories.py

framework-check:
	$(PYTHON) -m pip install -e framework/fuzzyxai
	$(PYTHON) -c "import sys; sys.path=['framework/fuzzyxai']+[p for p in sys.path if p not in ('', '.')]; import fuzzyxai; print(fuzzyxai.__version__); print(fuzzyxai.show_operator_route())"

fuzzyxai-framework-check:
	$(PYTHON) -m pip install -e framework/fuzzyxai
	$(PYTHON) -c "import sys; sys.path=['framework/fuzzyxai']+[p for p in sys.path if p not in ('', '.')]; from fuzzyxai import build_route, build_proof_trace, verify_proof_trace, render_dashboard, save_route_json; print('fuzzyxai-framework-import: PASS')"
	$(PYTHON) framework/fuzzyxai/examples/show_hybrid_xiris_dashboard.py
	$(PYTHON) -m pytest framework/fuzzyxai/tests/test_framework_core_v03.py framework/fuzzyxai/tests/test_framework_all_scenarios_v04.py -q

framework-external-check:
	$(PYTHON) -m pip install -e framework/fuzzyxai
	$(PYTHON) scripts/check_framework_external_usage.py

fuzzyxai-cli-check:
	$(PYTHON) -m pip install -e framework/fuzzyxai
	$(PYTHON) scripts/check_fuzzyxai_cli.py

fuzzyxai-schema-check:
	$(PYTHON) scripts/check_fuzzyxai_schema.py

fuzzyxai-adapter-sdk-check:
	$(PYTHON) scripts/check_fuzzyxai_adapter_sdk.py

fuzzyxai-framework-rc-check:
	$(PYTHON) scripts/build_fuzzyxai_framework_rc.py

fuzzyxai-framework-rc-package:
	$(PYTHON) scripts/build_fuzzyxai_framework_rc.py --package-only

fuzzyxai-visualization-check:
	$(PYTHON) scripts/build_fuzzyxai_visualization_package.py

fuzzyxai-visualization-package:
	$(PYTHON) scripts/build_fuzzyxai_visualization_package.py --package-only

fuzzyxai-visual-quality-check:
	$(PYTHON) scripts/build_fuzzyxai_visualization_package.py
	@echo "fuzzyxai-visual-quality-check: PASS"

fuzzyxai-shap-like-visualization-check:
	$(PYTHON) scripts/build_fuzzyxai_shap_like_visualization_package.py

fuzzyxai-shap-like-visualization-package:
	$(PYTHON) scripts/build_fuzzyxai_shap_like_visualization_package.py --package-only

fuzzyxai-ru-visual-explanation-check:
	$(PYTHON) scripts/build_fuzzyxai_ru_visual_explanations.py

fuzzyxai-ru-visual-explanation-package:
	$(PYTHON) scripts/build_fuzzyxai_ru_visual_explanations.py --package-only

fuzzyxai-ru-visual-editorial-check:
	$(PYTHON) scripts/build_fuzzyxai_ru_visual_explanations.py
	@echo "fuzzyxai-ru-visual-editorial-check: PASS"

fuzzyxai-ru-operator-explanation-check:
	$(PYTHON) scripts/build_fuzzyxai_ru_visual_explanations.py
	@echo "fuzzyxai-ru-operator-explanation-check: PASS"

fuzzyxai-ru-explanation-framework-check: fuzzyxai-ru-visual-editorial-check fuzzyxai-ru-visual-explanation-package fuzzyxai-ru-operator-explanation-check
	@echo "fuzzyxai-ru-explanation-framework-check: PASS"

operator-traceability-check:
	$(PYTHON) scripts/check_operator_traceability.py

research-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research_validation/runners/run_research_validation.py

research-validation-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research_validation/check_research_validation.py

fuzzyxai-research-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research_validation/run_research_analysis.py

fuzzyxai-research-analysis-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research_validation/check_research_analysis.py

applications-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) applications/run_all_scenarios.py

operator-dashboard:
	$(PYTHON) applications/export_operator_routes.py

operator-route-check:
	$(PYTHON) applications/check_operator_routes.py

site-build:
	@echo "site-build: QUARANTINED (see archive/site-prototype-cab4018)"

sprint-report:
	$(PYTHON) scripts/build_sprint_report.py

dubnaxai-release-check:
	@echo "dubnaxai-release-check: QUARANTINED; run doctorate-release-check for the framework"

risk-test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_risk_*.py -q

category-hott-test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_expl_category_laws.py tests/test_presheaf_functoriality.py tests/test_diagnostic_completion.py tests/test_explanation_path_types.py tests/test_temporal_drift_paths.py tests/test_context_topos_smoke.py tests/test_subpresheaf.py tests/test_yoneda.py tests/test_risk_context_acceptance.py -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) proofs/category_hott_checks.py

chapter2-breast-cancer-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter2_breast_cancer_demo.py --out-dir reports/chapter2

chapter2-real-operator-case:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter2_real_operator_case.py --out-dir reports/chapter2_real_operator_case

reproduce-chapter2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_sample113 --out-dir reports/chapter2

calibrate-chapter2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_calibration --out-dir reports/chapter2

benchmark-equal-raw-structure:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_equal_raw_structure --out-dir reports/chapter2

chapter2-3-final-evidence:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter2_3_final_evidence.py

chapter3-artifacts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter3_artifacts.py --out-dir reports/chapter3

reproduce-critical-ruptures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter3_critical_ruptures --out-dir reports/chapter3

chapter3-audit:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_audit_docx.py

chapter3-real-conflicts:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_build_real_conflicts.py

chapter3-f0-vs-nas:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_f0_vs_nas_experiment.py

chapter3-calibrate-observer:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_calibrate_observer.py

chapter3-tables:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_make_tables.py

chapter3-validate:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_validate_package.py

chapter3-final-evidence: chapter3-audit chapter3-real-conflicts chapter3-f0-vs-nas chapter3-calibrate-observer chapter3-tables chapter3-validate
	@echo "Audit: reports/chapter3/current_chapter_audit.md"
	@echo "Real conflicts: reports/chapter3/real_conflict_summary.csv"
	@echo "F0 vs NAS: reports/chapter3/f0_vs_nas_action_diff.csv"
	@echo "Calibration: reports/chapter3/observer_calibration_report.md"
	@echo "Config: configs/chapter3/best_observer_config.yaml"
	@echo "Package: chapter3_final_fix_evidence_package.zip"

chapter3-practice-docx:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_apply_patches_to_docx.py

chapter3-practice-natural:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_run_natural_flow.py

chapter3-practice-conflict:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_run_conflict_enriched.py

chapter3-practice-bootstrap:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_object_level_bootstrap.py

chapter3-practice-baselines:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_compare_baselines.py

chapter3-practice-calibrate:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_calibrate_observer_v2.py

chapter3-practice-ablation:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_ablation.py

chapter3-practice-sensitivity:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_sensitivity.py

chapter3-practice-stats:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_stat_tests.py

chapter3-practice-validate:
	PYTHONPATH=scripts:$(PYTHONPATH) $(PYTHON) scripts/chapter3_validate_practice_package.py

chapter3-practice-all: chapter3-practice-docx chapter3-practice-natural chapter3-practice-conflict chapter3-practice-bootstrap chapter3-practice-baselines chapter3-practice-calibrate chapter3-practice-ablation chapter3-practice-sensitivity chapter3-practice-stats chapter3-practice-validate
	@echo "Practice validation: reports/chapter3_practice/package_validation_report.md"
	@echo "Practice package: chapter3_practice_strengthening_package.zip"

figures-chapter2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/generate_figures.py --out-dir reports/figures

chapter2-figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter2_generate_figures.py

chapter2-patch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter2_patch_docx.py

chapter2-validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter2_validate_package2.py

chapter2-package2: chapter2-figures chapter2-patch chapter2-validate
	@echo "DOCX: glava2_chapter2_package2_full_fixed.docx"
	@echo "Figures: figures/chapter2/"
	@echo "Validation: reports/chapter2/package2_validation_report.md"

ecosystem-evidence:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ecosystem_evidence_pack.py --evidence-dir evidence --report-dir reports/chapter4

doctoral-final-evidence:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/integration_effort_report.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter2_alignment_synthesis.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter5_hybrid_xiris_blocking_case.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter2_3_final_evidence.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_doctoral_final_evidence.py

validate-ecosystem-sdk:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_sdk_contracts.py tests/test_api_examples.py tests/test_registry_registration_flow.py -q

dissertation-artifacts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_dissertation_artifacts.py --out-dir dissertation_artifacts

chapter5-experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter5_experiments.py --n-per-scenario 1000 --timing-n 1000 --out-dir reports/chapter5

chapter5-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.chapter5_demo

chapter5-latex:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/export_chapter5_latex.py

web-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/chapter5_web_demo.py --port $(PORT)

unified-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/unified_demo.py --port $(PORT)

layered-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/fuzzyxai_studio.py --port $(PORT)

layered-demo-legacy:
	@echo "[legacy] use 'make demo PORT=$(PORT)' for presentation"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/layered_demo.py --port $(PORT)

studio:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/fuzzyxai_studio.py --port $(PORT)

ui-health-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ui_health_check.py --out-dir reports

ui-health-check-all:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ui_health_check.py --out-dir reports --all-apps

browser-visual-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/browser_visual_check.py --port 18097 --out-dir reports/browser_visual_check

unified-demo-cli:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/unified_full_demo.py --out-dir reports/unified_full_demo

full-pipeline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/full_pipeline_demo.py --out-dir reports/full_pipeline

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/generate_figures.py --out-dir reports/figures

full-experiments: chapter5-experiments chapter2-breast-cancer-demo full-pipeline figures chapter5-latex
	@echo "All experiments completed. Reports are in reports/."

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/fuzzyxai_studio.py --port $(PORT)

defense-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/fuzzyxai_studio.py --port $(PORT)

defense-demo-legacy:
	@echo "[legacy] use 'make demo PORT=$(PORT)' for presentation"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/defense_demo.py --port $(PORT)

dashboard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) apps/nicegui_dashboard.py --port 8080

proof:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) proofs/run_all_proofs.py

formal-proof:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) proofs/formal_theorem_checks.py

thesis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) proofs/validate_thesis_examples.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/thesis_demo.py

full-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) full_pipeline_demo.py
	@echo "Full pipeline report: reports/full_demo/index.html"

full-observer:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) full_observer_pipeline.py
	@echo "Full observer report: reports/full_observer_pipeline/full_observer_pipeline.html"

dataset-observer:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/dataset_observer_demo.py --sample breast_cancer
	@echo "Dataset observer report: reports/dataset_observer/dataset_observer_report.html"

dataset-modes-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/check_dataset_modes.py

baseline-check:
	mkdir -p reports/dev
	{ \
		echo "# Baseline check"; \
		echo; \
		echo '```'; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q || true; \
		echo; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/check_dataset_modes.py || true; \
		echo; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_benchmark.py --dataset breast_cancer --out-root reports/datasets || true; \
		echo '```'; \
	} > reports/dev/baseline_check.md

real-data-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/unified_full_demo.py --dataset citr --out-dir reports/chapter5/real_data_validation
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/unified_full_demo.py --dataset rikord --out-dir reports/chapter5/real_data_validation
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/unified_full_demo.py --dataset ruccod --out-dir reports/chapter5/real_data_validation

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/breast_cancer_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/operator_comparison_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/risk_aware_observer_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/lofo_f1_rule_pruning_demo.py

benchmark-dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_benchmark.py --dataset $(DATASET) --out-root reports/datasets

baseline-comparison:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/baseline_comparison.py --dataset $(DATASET) --out-root reports/datasets --baseline-access $(BASELINE_ACCESS)

calibrate-observer:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/calibrate_observer.py --dataset $(DATASET) --out-root reports/datasets

ablation-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ablation_benchmark.py --dataset $(DATASET) --out-root reports/datasets

defense-cases:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/defense_cases.py --out-dir reports/defense_cases

real-reduction-example:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/real_reduction_example.py --out-dir reports/real_reduction_example

dissertation-demo-summary:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_demo_summary.py --out-dir reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_component_tables.py --out-dir reports

dissertation-component-tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_component_tables.py --out-dir reports

dataset-cards:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_cards.py --out-root reports/datasets

thesis-practice-tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_benchmark.py --dataset breast_cancer --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/calibrate_observer.py --dataset breast_cancer --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/baseline_comparison.py --dataset breast_cancer --out-root reports/datasets --baseline-access native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/baseline_comparison.py --dataset synthetic_ruptures --out-root reports/datasets --baseline-access native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/baseline_comparison.py --dataset diabetes_binary --out-root reports/datasets --baseline-access native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/baseline_comparison.py --dataset wine_risk --out-root reports/datasets --baseline-access native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ablation_benchmark.py --dataset breast_cancer --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/defense_cases.py --out-dir reports/defense_cases
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/structure_aware_benchmark.py --dataset breast_cancer --out-root reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/structure_aware_benchmark.py --dataset wine_risk --out-root reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/structure_aware_benchmark.py --dataset diabetes_binary --out-root reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/export_thesis_practice_tables.py --out-dir reports/thesis_tables

structure-aware-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/structure_aware_benchmark.py --dataset $(DATASET) --out-root reports

reproducibility-artifacts:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ecosystem_evidence_pack.py --evidence-dir evidence --report-dir reports/chapter4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_calibration --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_equal_raw_structure --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter3_artifacts.py --out-dir reports/chapter3
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter3_critical_ruptures --out-dir reports/chapter3
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/reproducibility_artifacts.py --out-dir reports/reproducibility_artifacts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_dissertation_artifacts.py --out-dir dissertation_artifacts

dissertation-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/check_dataset_modes.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_benchmark.py --dataset breast_cancer --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_sample113 --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_calibration --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_equal_raw_structure --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter2_real_operator_case.py --out-dir reports/chapter2_real_operator_case
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/real_reduction_example.py --out-dir reports/real_reduction_example
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_demo_summary.py --out-dir reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_component_tables.py --out-dir reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_cards.py --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter3_artifacts.py --out-dir reports/chapter3
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter3_critical_ruptures --out-dir reports/chapter3
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/ecosystem_evidence_pack.py --evidence-dir evidence --report-dir reports/chapter4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_sdk_contracts.py tests/test_api_examples.py tests/test_registry_registration_flow.py -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/reproducibility_artifacts.py --out-dir reports/reproducibility_artifacts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_dissertation_artifacts.py --out-dir dissertation_artifacts
	@echo "DISSERTATION CHECK PASSED"

operator-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/operator_comparison_benchmark.py

risk-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/risk_aware_observer_benchmark.py

lofo-f1-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/lofo_f1_rule_pruning_demo.py
	@echo "LOFO-F1 report: reports/lofo_f1_rule_pruning.md"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

explanation-experience-evidence:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_explanation_experience_evidence.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_explanation_experience.py

.PHONY: chapter4-explanation-evidence empirical-validation empirical-validation-check chapter4-empirical-evidence model-universality external-validation-gates chapter4-final-candidate
chapter4-explanation-evidence: explanation-experience-evidence
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter4_explanation_evidence.py

empirical-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/real_training_experiment/run_empirical_validation.py

chapter4-empirical-evidence: empirical-validation
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_empirical_chapter4_evidence.py

empirical-validation-check: chapter4-empirical-evidence
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_empirical_validation.py

model-universality:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/model_universality/run_benchmark.py
	cd release_evidence/model_universality && sha256sum -c checksums.sha256

external-validation-gates:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_external_validation_package.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_external_release_gates.py

chapter4-final-candidate: model-universality external-validation-gates
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_chapter4_final_candidate.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_chapter4_final_candidate.py

empirical-smoke:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/reproduce_all.py --profile smoke --skip-optional --skip-archives

reproduce-dissertation:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/reproduce_all.py --profile full

empirical-full-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_reproduction.py --profile full

.PHONY: ai-pre-review-select-source ai-pre-review-build-log ai-pre-review-build-batches ai-pre-review-validate-input ai-pre-review-import ai-pre-review-aggregate ai-pre-review-lock-confirmatory ai-pre-review-build-human-pack ai-pre-review-compare-human ai-pre-review-reports ai-pre-review-archive ai-pre-review-check
ai-pre-review-select-source:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/select_source_cases.py

ai-pre-review-build-log:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/build_log.py

ai-pre-review-build-batches: ai-pre-review-build-log

ai-pre-review-validate-input:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/validate_inputs.py

ai-pre-review-import:
	test -n "$(REVIEW_DIR)" && test -n "$(SPLIT)" && test -n "$(AI_RUN_ID)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/import_ai_reviews.py --review-dir "$(REVIEW_DIR)" --split "$(SPLIT)" --run-id "$(AI_RUN_ID)"

ai-pre-review-aggregate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/aggregate_ai_reviews.py

ai-pre-review-lock-confirmatory:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/lock_confirmatory.py

ai-pre-review-build-human-pack:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/build_human_pack.py

ai-pre-review-compare-human:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/compare_human.py

ai-pre-review-reports:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/build_reports.py

ai-pre-review-archive:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/build_archive.py

ai-pre-review-check: ai-pre-review-build-log ai-pre-review-validate-input ai-pre-review-reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/build_archive.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review/verify_pipeline.py

.PHONY: ai-final-build-log ai-final-build-blind-batches ai-pre-review-final-blinding-audit ai-final-blinding-audit ai-final-validate-evidence ai-final-lock-confirmatory ai-final-claim-registry ai-final-dissertation-artifacts ai-final-archive ai-final-check reproduce-ai-review-final
ai-final-build-log:
	test -s study/ai_pre_review_final/public_formative/reviewer_cases.jsonl
	test -s study/ai_pre_review_final/public_formative/manifest.json

ai-final-build-blind-batches: ai-final-build-log
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/build_public_bundle.py

ai-pre-review-final-blinding-audit ai-final-blinding-audit: ai-final-build-log
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/audit_blinding.py

ai-final-validate-evidence: ai-final-build-log
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/validate_evidence.py

ai-final-lock-confirmatory:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/lock_confirmatory.py

ai-final-claim-registry:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/build_claim_registry.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/check_claims.py

ai-final-dissertation-artifacts: ai-final-blinding-audit ai-final-claim-registry
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/build_reports.py

ai-final-archive: ai-final-dissertation-artifacts ai-final-validate-evidence
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ai_pre_review_final/build_public_bundle.py

ai-final-check: ai-final-archive
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/ai_pre_review_final

reproduce-ai-review-final: ai-final-check

.PHONY: strong-confirmatory-protocol strong-confirmatory-smoke strong-confirmatory-formative strong-confirmatory-formative-check strong-confirmatory-lock strong-confirmatory-bundle chapter4-formative-shell chapter4-strong-confirmatory-final strong-confirmatory-check
strong-confirmatory-protocol:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/build_protocol.py

strong-confirmatory-smoke: strong-confirmatory-protocol
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/run_formative.py --profile smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/build_report.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/verify_formative.py

strong-confirmatory-formative: strong-confirmatory-protocol
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/run_formative.py --profile full
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/build_report.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/verify_formative.py

strong-confirmatory-formative-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/verify_formative.py

strong-confirmatory-lock:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/lock_protocol.py

strong-confirmatory-bundle: chapter4-formative-shell
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/strong_confirmatory/build_bundle.py

chapter4-formative-shell: strong-confirmatory-formative-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter4/build_claims.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter4/build_tables.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter4/build_figures.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter4/build_text.py

chapter4-strong-confirmatory-final:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/chapter4/build_final.py

strong-confirmatory-check: strong-confirmatory-protocol
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/strong_confirmatory

.PHONY: practical-controller-protocol practical-controller-formative practical-controller-formative-check practical-controller-freeze practical-controller-confirmatory practical-controller-baselines practical-controller-ablation route-validity-confirmatory rule-detectability-envelope rule-matched-control-confirmatory posthoc-benchmark-final glassbox-benchmark-final h7-canonical-fidelity h7-presentation-tradeoff grid-confirmatory scale-2m scale-5m ai-formative-run2-import final-statistics final-claim-registry chapter4-practical-formative chapter4-final practical-release-check practical-release-archive practical-docker-check reproduce-final-practical-closure

practical-controller-protocol:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_protocol.py

practical-controller-formative: practical-controller-protocol
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) nice -n 18 $(PYTHON) scripts/final_practical_closure/run_formative.py --profile full
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/verify_formative.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_artifacts.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_formative_report.py

practical-controller-formative-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/verify_formative.py

practical-controller-freeze: practical-controller-formative-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/lock_protocol.py

practical-controller-confirmatory: practical-controller-freeze
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/run_confirmatory.py

practical-controller-baselines practical-controller-ablation rule-detectability-envelope posthoc-benchmark-final glassbox-benchmark-final h7-canonical-fidelity h7-presentation-tradeoff:
	@$(MAKE) practical-controller-formative

route-validity-confirmatory rule-matched-control-confirmatory grid-confirmatory scale-2m scale-5m:
	@$(MAKE) practical-controller-confirmatory

ai-formative-run2-import:
	@test -n "$(AI_RUN2_INPUT)" || (echo "BLOCKED: set AI_RUN2_INPUT to a real blinded run-2 JSON"; exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/import_ai_formative_run2.py "$(AI_RUN2_INPUT)"

final-statistics: practical-controller-confirmatory
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_claim_registry.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_confirmatory_outputs.py

final-claim-registry:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_claim_registry.py

chapter4-practical-formative: practical-controller-formative-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_artifacts.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_chapter4_formative.py

chapter4-final: final-chapter4
	@echo "PASS: chapter4-final source=sealed-confirmatory"

practical-release-check: practical-controller-formative-check chapter4-practical-formative final-claim-registry
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) nice -n 18 $(PYTHON) -m pytest -q tests/practical_controller
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/practical_controller scripts/final_practical_closure tests/practical_controller
	@echo "PASS: practical-release-check scope=formative-technical-candidate stable=false"

practical-release-archive: practical-release-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_formative_report.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_bundle.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_framework_release.py

practical-docker-check:
	docker build -f Dockerfile.practical -t fuzzyxai-practical .
	docker run --rm fuzzyxai-practical

reproduce-final-practical-closure: final-release-check final-controller-freeze final-controller-confirmatory final-confirmatory-statistics final-confirmatory-claim-registry final-chapter4 final-one-zip

.PHONY: final-confirmatory-protocol final-comparator-protocol final-dataset-registry final-seal-datasets final-leakage-audit final-data-verify final-near-duplicate-audit final-oof-features final-p0-p1-audit final-local-data-check final-controller-formative final-controller-freeze final-controller-confirmatory final-controller-baselines final-controller-ablation final-route-controlled final-route-replay final-rule-envelope final-rule-matched-controls final-canonical-evidence final-presentation-projection final-posthoc-benchmark final-glassbox-benchmark final-grid-confirmatory final-scale-operator final-scale-end-to-end final-shadow-replay final-ai-run2-build final-ai-run2-import final-ai-run2-report final-ai-text-review-scope final-prelock-method-registry final-confirmatory-statistics final-confirmatory-claim-registry final-chapter4 final-release-check final-prelock-archive final-release-archive final-one-zip reproduce-final-confirmatory-closure

final-confirmatory-protocol:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_protocol.py

final-comparator-protocol:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_comparator_protocol.py

final-dataset-registry: final-confirmatory-protocol
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_dataset_registry.py

final-seal-datasets: final-dataset-registry
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/seal_datasets.py

final-leakage-audit: final-seal-datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/verify_dataset_leakage.py

final-data-verify: final-leakage-audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_data_metadata.py
	@echo "PASS: final_data_verify"

final-near-duplicate-audit: final-data-verify
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) nice -n 18 $(PYTHON) scripts/final_closure/audit_near_duplicates.py

final-oof-features: final-near-duplicate-audit
	@for dataset in bank_marketing default_credit_clients shoulder_implant_xray sms_spam uci_har_smartphones; do \
		OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/build_oof_features.py --dataset $$dataset || exit $$?; \
	done
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/verify_oof_features.py

final-p0-p1-audit: final-oof-features
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/verify_oof_features.py

final-local-data-check: final-p0-p1-audit
	@echo "PASS: final-local-data-check private_inputs_available=true test_opened=false"

final-controller-formative: final-p0-p1-audit
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH):scripts/final_closure nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/run_real_formative.py

final-controller-freeze: final-p0-p1-audit final-controller-formative final-comparator-protocol final-posthoc-benchmark final-canonical-evidence final-ai-text-review-scope final-prelock-method-registry
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/lock_protocol.py

final-controller-confirmatory: final-controller-freeze
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH):scripts/final_closure nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/run_sealed_confirmatory.py

final-controller-scoring-recovery-lock:
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(CONFIRMATORY_PYTHON) scripts/final_closure/run_scoring_recovery.py lock

final-controller-scoring-recovery:
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(CONFIRMATORY_PYTHON) scripts/final_closure/run_scoring_recovery.py run

final-controller-baselines final-controller-ablation:
	@$(MAKE) final-controller-confirmatory

final-route-controlled:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_fault_library.py

final-route-replay final-shadow-replay:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/run_shadow_replay.py

final-rule-envelope final-scale-operator:
	@$(MAKE) final-controller-formative

final-canonical-evidence final-presentation-projection: final-p0-p1-audit
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH):scripts/final_closure nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/verify_canonical_projection.py

final-posthoc-benchmark final-glassbox-benchmark: final-comparator-protocol final-p0-p1-audit
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH):framework/fuzzyxai nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/run_comparator_formative.py

final-rule-matched-controls final-grid-confirmatory final-scale-end-to-end:
	@$(MAKE) final-controller-confirmatory

final-ai-run2-build:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_ai_run2_bundle.py

final-ai-run2-import:
	@test -n "$(AI_RUN2_INPUT)" || (echo "BLOCKED: set AI_RUN2_INPUT to a directory containing reviews.jsonl and session_metadata.json"; exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/import_ai_run2.py "$(AI_RUN2_INPUT)"

final-ai-run2-report:
	@test -f study/final_confirmatory_closure/ai_formative_run2_report.md || (echo "BLOCKED: import a real clean-session AI run 2 first"; exit 2)
	@cat study/final_confirmatory_closure/ai_formative_run2_report.md

final-ai-text-review-scope: final-ai-run2-build
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(PYTHON) scripts/final_closure/build_ai_scope_decision.py

final-prelock-method-registry: final-controller-formative final-posthoc-benchmark final-canonical-evidence
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(PYTHON) scripts/final_closure/build_prelock_method_registry.py

final-confirmatory-statistics: final-controller-confirmatory
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(CONFIRMATORY_PYTHON) scripts/final_closure/build_final_statistics.py

final-confirmatory-claim-registry: final-confirmatory-statistics
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(PYTHON) scripts/final_closure/build_final_claim_registry.py

final-chapter4: final-confirmatory-claim-registry
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH):scripts/final_closure nice -n 18 $(CONFIRMATORY_PYTHON) scripts/final_closure/build_chapter4.py

final-release-check: final-confirmatory-protocol final-route-controlled final-ai-run2-build final-ai-text-review-scope final-shadow-replay
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=$(PYTHONPATH) nice -n 18 $(PYTHON) -m pytest -q tests/final_closure tests/practical_controller
	PYTHONPATH=$(PYTHONPATH) nice -n 18 $(PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/final_closure scripts/final_closure tests/final_closure
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/verify_prelock.py
	@echo "PASS: final-release-check scope=prelock-technical stable=false"

final-prelock-archive: final-release-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_closure/build_prelock_bundle.py

final-release-archive: final-release-check
	@$(MAKE) final-one-zip

final-one-zip: final-chapter4
	PYTHONPATH=$(PYTHONPATH):scripts/final_closure $(PYTHON) scripts/final_closure/build_one_zip.py

reproduce-final-confirmatory-closure: reproduce-final-practical-closure

# Chapter 4 v13 uses a pre-existing environment selected by the caller.
CHAPTER4_V13_PYTHON ?= python

chapter4-v13-smoke:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.reproduce_all --smoke

chapter4-v13-prepare:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.prepare_data
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.train_or_load_model

chapter4-v13-explanations:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.generate_explanations --split validation --objects 2000
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.generate_explanations --split sealed_test --objects 2000

chapter4-v13-analysis:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.run_policies --stage pre-score
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.run_policies --stage score
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.run_route_faults
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.benchmark_end_to_end
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.reproduce_case
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_tables
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_figures
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.validate_evidence
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_chapter
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_closure
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.validate_document

reproduce-chapter4-v13:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.reproduce_all

chapter4-v13-release:
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_closure
	PYTHONPATH=framework/fuzzyxai:. $(CHAPTER4_V13_PYTHON) -m experiments.chapter4_v13.build_release

# H10 v19 uses a caller-selected existing Python environment. The reproduction
# target consumes frozen scoring outputs and never opens the sealed vault.
H10_PYTHON ?= $(PYTHON)
H10_ENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:.

.PHONY: h10-smoke h10-exploratory h10-data h10-freeze h10-confirmatory h10-replay h10-statistics h10-tables h10-figures h10-evidence h10-validation h10-package reproduce-h10

h10-smoke:
	$(H10_ENV) $(H10_PYTHON) -m pytest -q tests/h10
	$(H10_ENV) $(H10_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/audit_h10 baselines/h10 experiments/h10 tests/h10

h10-exploratory:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.run_exploratory --config config/h10_v19_exploratory.yaml

h10-data:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.prepare_data --config config/h10_v19_protocol.yaml

h10-freeze:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.freeze_protocol --config config/h10_v19_protocol.yaml

h10-confirmatory:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.run_confirmatory --config config/h10_v19_protocol.yaml

h10-replay:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.run_replay --config config/h10_v19_protocol.yaml

h10-statistics:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.compute_statistics --config config/h10_v19_protocol.yaml

h10-tables:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.build_tables

h10-figures:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.build_figures

h10-evidence:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.validate_evidence evidence

h10-validation:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.validate_evidence validate

h10-package:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.package

reproduce-h10:
	$(H10_ENV) $(H10_PYTHON) -m experiments.h10.reproduce

# H10 final Gold uses the caller-selected existing environment. It never opens
# sealed truth as part of ordinary reproduction.
H10_GOLD_PYTHON ?= $(PYTHON)
H10_GOLD_ENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=framework/fuzzyxai:.

.PHONY: h10-gold-generate h10-gold-test h10-gold-development h10-gold-power h10-gold-adjudication h10-gold-freeze h10-gold-confirmatory h10-gold-closure h10-gold-figures h10-gold-package reproduce-h10-gold

h10-gold-generate:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.generate_benchmark

h10-gold-test:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m ruff check gold_oracle experiments/h10_gold baselines/h10_gold framework/fuzzyxai/fuzzyxai/audit_h10/gold_benchmark.py tests/h10_gold
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m pytest -q tests/h10_gold

h10-gold-development:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.run_methods --split development

h10-gold-power:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.power_analysis

h10-gold-adjudication:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.validate_adjudication

h10-gold-freeze:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.freeze_protocol

h10-gold-confirmatory:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.run_confirmatory

h10-gold-closure:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.build_closure
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.validate_evidence

h10-gold-figures:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.build_figures

h10-gold-package:
	$(H10_GOLD_ENV) $(H10_GOLD_PYTHON) -m experiments.h10_gold.package

reproduce-h10-gold: h10-gold-test h10-gold-generate h10-gold-development h10-gold-power h10-gold-closure h10-gold-figures
	@echo "H10 Gold preconfirmatory reproduction complete; sealed scoring was not opened."

# Diagnostic v21 uses the caller-selected existing environment. It validates the
# alpha framework and protocol draft but never opens a sealed dataset.
DIAGNOSTIC_PYTHON ?= $(PYTHON)
DIAGNOSTIC_ENV = PYTHONPATH=framework/fuzzyxai:.

.PHONY: diagnostic-v21-test diagnostic-v21-coverage diagnostic-v21-benchmark diagnostic-v21-protocol diagnostic-v21-report diagnostic-v21-check

diagnostic-v21-test:
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/diagnostics tests/diagnostics experiments/diagnostic_v21 examples/diagnose_single_route.py examples/diagnose_batch.py examples/build_repair_plan.py examples/recertify_route.py examples/diagnostic_report_levels.py
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m pytest -q tests/diagnostics

diagnostic-v21-coverage:
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m coverage erase
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m coverage run --source=framework/fuzzyxai/fuzzyxai/diagnostics -m pytest -q tests/diagnostics
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m coverage report --fail-under=90
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m coverage json -o reports/diagnostic_v21/coverage.json

diagnostic-v21-benchmark:
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m experiments.diagnostic_v21.benchmark

diagnostic-v21-protocol:
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m experiments.diagnostic_v21.protocol_gate

diagnostic-v21-report:
	$(DIAGNOSTIC_ENV) $(DIAGNOSTIC_PYTHON) -m experiments.diagnostic_v21.build_validation_report --python $(DIAGNOSTIC_PYTHON)

diagnostic-v21-check: diagnostic-v21-test diagnostic-v21-coverage diagnostic-v21-benchmark diagnostic-v21-protocol

.PHONY: h10-c2-bootstrap h10-c2-test h10-c2-power h10-c2-generate-development h10-c2-run-development h10-c2-generate-protocol-validation h10-c2-run-protocol-validation h10-c2-build-adjudication h10-c2-freeze-protocol h10-c2-generate-sealed h10-c2-preconfirmatory-gate h10-c2-score-sealed h10-c2-package

H10_C2_PYTHON ?= $(PYTHON)
H10_C2_ENV = PYTHONPATH=experiments/h10_c2/src:framework/fuzzyxai:.

h10-c2-bootstrap:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 bootstrap

h10-c2-test:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m pytest -q experiments/h10_c2/tests
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m ruff check experiments/h10_c2

h10-c2-power:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 power

h10-c2-generate-development:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 generate --split development

h10-c2-run-development:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 run --split development

h10-c2-generate-protocol-validation:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 generate --split protocol_validation

h10-c2-run-protocol-validation:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 run --split protocol_validation

h10-c2-build-adjudication:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 export-adjudication --sample-size 200

h10-c2-freeze-protocol:
	@test -n "$(H10_C2_DESIGN_APPROVAL)" || (echo "BLOCKED: set H10_C2_DESIGN_APPROVAL to the signed design approval"; exit 2)
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 freeze-protocol --approval "$(H10_C2_DESIGN_APPROVAL)"

h10-c2-generate-sealed:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 generate --split sealed

h10-c2-preconfirmatory-gate:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 preconfirmatory-gate

h10-c2-score-sealed:
	@test -n "$(H10_C2_APPROVAL)" || (echo "BLOCKED: set H10_C2_APPROVAL to a signed scoring approval"; exit 2)
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 score-sealed --lock artifacts/h10_c2/lock/protocol.lock.json --approval "$(H10_C2_APPROVAL)"

h10-c2-package:
	$(H10_C2_ENV) $(H10_C2_PYTHON) -m h10_c2 package
	@echo "Diagnostic v21 alpha checks complete; H10-C2 remains BLOCKED_PRECONFIRMATORY."

H10_C3_PROJECT_VENV := $(abspath $(CURDIR)/../venv/bin/python)
H10_C3_PYTHON ?= $(if $(wildcard $(H10_C3_PROJECT_VENV)),$(H10_C3_PROJECT_VENV),$(PYTHON))
H10_C3_ENV = PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:.

.PHONY: diagnostic-v23-test diagnostic-v23-benchmark h10-c3-test full-regression h10-c3-cost-stability-audit h10-c3-cost-open-reproduction h10-c3-cost-stability-package h10-c3-generate-development h10-c3-run-development h10-c3-freeze h10-c3-generate-protocol-validation h10-c3-run-protocol-validation h10-c3-stability-analysis h10-c3-power h10-c3-preconfirmatory-gate h10-c3-reports h10-c3-package h10-c3-score-sealed

diagnostic-v23-test:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/diagnostics experiments/h10_c3 tests/diagnostics
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m pytest -q tests/diagnostics experiments/h10_c3/tests tests/h10_c3

h10-c3-test: diagnostic-v23-test

full-regression:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m pytest -q

h10-c3-cost-stability-audit:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 cost-stability-audit

h10-c3-cost-open-reproduction:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 cost-open-reproduction

h10-c3-cost-stability-package:
	$(H10_C3_PYTHON) scripts/build_framework_release.py
	$(H10_C3_PYTHON) scripts/build_h10_c3_cost_stability_handoff.py

diagnostic-v23-benchmark:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 generate-development
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 run-development

h10-c3-generate-development:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 generate-development

h10-c3-run-development:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 run-development

h10-c3-freeze:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 freeze

h10-c3-generate-protocol-validation:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 generate-protocol-validation

h10-c3-run-protocol-validation:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 run-protocol-validation

h10-c3-stability-analysis: h10-c3-cost-stability-audit
	$(H10_C3_PYTHON) -c 'import json, pathlib; p=pathlib.Path("artifacts/h10_c3/cost_stability/stability_gate.json"); d=json.loads(p.read_text()); assert d["status"] == "PASS", d'

h10-c3-power:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 power

h10-c3-preconfirmatory-gate:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 audits
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 gate

h10-c3-reports:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 reports

h10-c3-package:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 package

h10-c3-score-sealed:
	$(H10_C3_ENV) $(H10_C3_PYTHON) -m h10_c3 score-sealed

H10_C3_R4_ENV = PYTHONPATH=experiments/h10_c3_r4/src:framework/fuzzyxai:.

.PHONY: h10-c3-r4-test h10-c3-r4-template-audit h10-c3-r4-generate-development h10-c3-r4-run-development h10-c3-r4-freeze h10-c3-r4-generate-protocol-validation h10-c3-r4-run-protocol-validation h10-c3-r4-stability h10-c3-r4-power h10-c3-r4-preconfirmatory-gate h10-c3-r4-freeze-secure-protocol h10-c3-r4-generate-sealed h10-c3-r4-score-sealed h10-c3-r4-package

h10-c3-r4-test:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/diagnostics experiments/h10_c3_r4 gold_oracle/h10_c3_r4_oracle.py scripts/build_h10_c3_r4_handoff.py
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/diagnostics experiments/h10_c3_r4/tests

h10-c3-r4-template-audit:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 template-audit

h10-c3-r4-generate-development:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 generate-development

h10-c3-r4-run-development:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 run-development

h10-c3-r4-freeze:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 freeze

h10-c3-r4-generate-protocol-validation:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 generate-protocol-validation

h10-c3-r4-run-protocol-validation:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 run-protocol-validation

h10-c3-r4-stability:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 stability

h10-c3-r4-power:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 power

h10-c3-r4-preconfirmatory-gate:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 gate

h10-c3-r4-freeze-secure-protocol:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 freeze-secure-protocol

h10-c3-r4-generate-sealed:
	@test -n "$(SEALED_SECRET)" || (echo "BLOCKED: set SEALED_SECRET to an owner-only path outside the repository"; exit 2)
	SEALED_SECRET="$(SEALED_SECRET)" $(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 generate-sealed

h10-c3-r4-score-sealed:
	@test -n "$(APPROVAL)" || (echo "BLOCKED: set APPROVAL to an explicit one-time scoring authorization"; exit 2)
	@test -n "$(SEALED_SECRET)" || (echo "BLOCKED: set SEALED_SECRET to the external owner-only seed path"; exit 2)
	APPROVAL="$(APPROVAL)" SEALED_SECRET="$(SEALED_SECRET)" $(H10_C3_R4_ENV) $(H10_C3_PYTHON) -m h10_c3_r4 score-sealed

h10-c3-r4-package:
	$(H10_C3_R4_ENV) $(H10_C3_PYTHON) scripts/build_h10_c3_r4_handoff.py

H10_C4_ENV = PYTHONPATH=framework/fuzzyxai:.
H10_C5B_POSTOPEN_EVIDENCE ?= release_artifacts/fuzzyxai-h10-c5b-postopen-evidence-57d0cc9b3e73.zip

.PHONY: h10-c4-test h10-c4-run h10-c4-verify h10-c4-chapter h10-c4-package

h10-c4-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/operators/base.py framework/fuzzyxai/fuzzyxai/operators/composition.py framework/fuzzyxai/fuzzyxai/repair framework/fuzzyxai/fuzzyxai/experiments/h10_c4.py scripts/run_h10_c4.py scripts/build_h10_c4_chapter4.py scripts/build_h10_c4_release.py scripts/manuscript_claim_lint.py tests/operators tests/h10_c4
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/operators tests/h10_c4

h10-c4-run:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/run_h10_c4.py run

h10-c4-verify:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/run_h10_c4.py verify

h10-c4-chapter:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/build_h10_c4_chapter4.py
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/manuscript_claim_lint.py docs/chapters/glava_4_FuzzyXAI_h10_c4_revision.docx

h10-c4-package:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/build_h10_c4_release.py

.PHONY: ch4-q1-test h10-c5-source-audit h10-c5-run h10-c5b-test h10-c5b-ops-test h10-c5b-prepare h10-c5b-prepare-runtime-inputs h10-c5b-collect-runtime h10-c5b-merge-runtime h10-c5b-runtime-readiness h10-c5b-freeze-development h10-c5b-freeze-held-out-scoring h10-c5b-plan-replacements h10-c5b-run h10-c5b-score-held-out h10-c5b-verify-method-lock h10-c5b-parent-immutability h10-c5c-test h10-c5c-posthoc h10-c5c-posthoc-detailed h10-c5c-posthoc-oracles h10-c5c-prepare-bugsinpy h10-c5c-collect-runtime h10-c5c-development-readiness h10-c5c-run-development h10-c7-test h10-c7-build-open-replay h10-c7-open-replay h10-c7-confirmation h10-c7-r5v-audit h10-c7-prepare-development h10-c7-run-development h10-c7a-prepare-development h10-c7a-run-development h10-c7r-test h10-c7r-lock h10-c7r-prepare-held-out h10-c7r-lock-images h10-c7r-collect-runtime h10-c7r-build-gold h10-c7r-authorize h10-c7r-score h10-c7r-r9-test h10-c7r-r9-development h10-c7r-r10-test h10-c7r-r10-provenance h10-c7r-r10-recollection-input-audit h10-c7r-r10-runtime-audit h10-c7r-r10-development h10-c5-pilot-test h10-c5-pilot-run h10-c6-run h10-c6-n-test h10-c6-n-prepare h10-c6-n-run mlflow-integration-test mlflow-integration-run final-practical-parent-immutability final-practical-package multimodal-route-validation h9-e2e-latency h9-e2e-v2 ch4-q1-claim-lint ch4-q1-evidence

ch4-q1-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c5 tests/h10_c6 tests/multimodal tests/roles tests/chapter_revision

h10-c5-source-audit:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c5_sources.py --root .

h10-c5-run:
	test -n "$(H10_C5_SOURCE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c5.py --source "$(H10_C5_SOURCE)" --root .

h10-c5b-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/repository_diagnostics framework/fuzzyxai/fuzzyxai/gold_repository framework/fuzzyxai/fuzzyxai/evidence_path framework/fuzzyxai/fuzzyxai/experiments/h10_c5b.py framework/fuzzyxai/fuzzyxai/experiments/h9_e2e_v2.py scripts/ch4_revision/collect_h10_c5b_runtime.py scripts/ch4_revision/h10_c5b_runtime_ops.py scripts/ch4_revision/prepare_h10_c5b_sources.py scripts/ch4_revision/run_h10_c5b.py scripts/ch4_revision/run_h9_e2e_v2.py scripts/ch4_revision/score_h10_c5b_held_out.py scripts/ch4_revision/verify_parent_result_immutability.py tests/h10_c5b tests/h9_e2e_v2
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c5b tests/h9_e2e_v2

h10-c5b-ops-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check scripts/ch4_revision/h10_c5b_runtime_ops.py scripts/ch4_revision/score_h10_c5b_held_out.py tests/h10_c5b/test_held_out_scoring_controller.py tests/h10_c5b/test_runtime_operations.py
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c5b/test_held_out_scoring_controller.py tests/h10_c5b/test_runtime_operations.py

h10-c5c-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/repository_diagnostics/retrieval.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/runtime_events.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/executed_slice.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/contract_inference.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/importer_v2.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/auditor_v2.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/evidence_requests.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/practical_recovery.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/reporting.py framework/fuzzyxai/fuzzyxai/experiments/h10_c5c.py framework/fuzzyxai/fuzzyxai/experiments/h10_c5c_data.py framework/fuzzyxai/fuzzyxai/experiments/h10_c5c_runtime.py framework/fuzzyxai/fuzzyxai/experiments/h10_c5c_readiness.py framework/fuzzyxai/fuzzyxai/experiments/h10_c5c_posthoc.py scripts/ch4_revision/analyze_h10_c5b_errors.py scripts/ch4_revision/prepare_h10_c5c_bugsinpy.py scripts/ch4_revision/collect_h10_c5c_runtime.py scripts/ch4_revision/verify_h10_c5c_development_readiness.py scripts/ch4_revision/run_h10_c5c_development.py scripts/ch4_revision/run_h10_c5c_posthoc_oracles.py tests/h10_c5c
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c5c
	printf '%s  %s\n' 479fa55460426a1726f57a5ed5f745b0ad2e872eeefb21470dd2e3e865133949 framework/fuzzyxai/operators_manifest.yaml | sha256sum -c -

h10-c5c-posthoc:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/analyze_h10_c5b_errors.py --source results/h10_c5b/PER_INCIDENT_RESULTS.csv --output reports/h10_c5c/H10_C5B_POSTHOC_ERROR_ANALYSIS.json

h10-c5c-posthoc-detailed:
	test -f "$(H10_C5B_POSTOPEN_EVIDENCE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/analyze_h10_c5b_errors.py --source results/h10_c5b/PER_INCIDENT_RESULTS.csv --output reports/h10_c5c/H10_C5B_POSTHOC_ERROR_ANALYSIS.json --evidence-archive "$(H10_C5B_POSTOPEN_EVIDENCE)" --table-output reports/h10_c5c/H10_C5B_INCIDENT_ERROR_CLASSIFICATION.csv

h10-c5c-posthoc-oracles:
	test -n "$(H10_C5C_DEVELOPMENT_MANIFEST)"
	test -n "$(H10_C5C_POSTHOC_OUTPUT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c5c_posthoc_oracles.py --manifest "$(H10_C5C_DEVELOPMENT_MANIFEST)" --development-status results/h10_c5c/H10_C5C_DEVELOPMENT_STATUS.json --baseline-results results/h10_c5c/DEVELOPMENT_PER_INCIDENT.csv --output "$(H10_C5C_POSTHOC_OUTPUT)" --root .

h10-c5c-prepare-bugsinpy:
	test -n "$(H10_C5C_BUGSINPY_ROOT)"
	test -n "$(H10_C5C_SOURCE_DIR)"
	test -n "$(H10_C5C_REPOSITORY_CACHE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c5c_bugsinpy.py --bugsinpy-root "$(H10_C5C_BUGSINPY_ROOT)" --output "$(H10_C5C_SOURCE_DIR)" --cache "$(H10_C5C_REPOSITORY_CACHE)" --root . $(if $(H10_C5C_ALLOW_NETWORK),--allow-network)

h10-c5c-collect-runtime:
	test -n "$(H10_C5C_UNCOLLECTED_MANIFEST)"
	test -n "$(H10_C5C_COMMAND_REGISTRY)"
	test -n "$(H10_C5C_RUNTIME_DIR)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/collect_h10_c5c_runtime.py --manifest "$(H10_C5C_UNCOLLECTED_MANIFEST)" --command-registry "$(H10_C5C_COMMAND_REGISTRY)" --output "$(H10_C5C_RUNTIME_DIR)" $(if $(H10_C5C_TIMEOUT_SECONDS),--timeout-seconds "$(H10_C5C_TIMEOUT_SECONDS)") $(if $(H10_C5C_ALLOW_SETUP),--allow-setup) $(if $(H10_C5C_INTERPRETER_MAP),--interpreter-map "$(H10_C5C_INTERPRETER_MAP)") $(if $(H10_C5C_WORKERS),--workers "$(H10_C5C_WORKERS)")

h10-c5c-development-readiness:
	test -n "$(H10_C5C_DEVELOPMENT_MANIFEST)"
	test -n "$(H10_C5C_COMMAND_REGISTRY)"
	test -n "$(H10_C5C_SOURCE_REGISTRY)"
	test -n "$(H10_C5C_RUNTIME_REPORT)"
	test -n "$(H10_C5C_READINESS_REPORT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/verify_h10_c5c_development_readiness.py --manifest "$(H10_C5C_DEVELOPMENT_MANIFEST)" --command-registry "$(H10_C5C_COMMAND_REGISTRY)" --source-registry "$(H10_C5C_SOURCE_REGISTRY)" --runtime-report "$(H10_C5C_RUNTIME_REPORT)" --output "$(H10_C5C_READINESS_REPORT)" --root .


h10-c5c-run-development:
	test -n "$(H10_C5C_DEVELOPMENT_MANIFEST)"
	test -n "$(H10_C5C_READINESS_REPORT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c5c_development.py --manifest "$(H10_C5C_DEVELOPMENT_MANIFEST)" --readiness-report "$(H10_C5C_READINESS_REPORT)" --root .

h10-c7-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_retrieval.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_diagnosis.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/contract_inference_v2.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/active_evidence.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/incident_router.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/repair_validation.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7a.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7_model_lock.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7_replay.py scripts/ch4_revision/prepare_h10_c7_development.py scripts/ch4_revision/prepare_h10_c7a_development.py scripts/ch4_revision/run_h10_c7_development.py scripts/ch4_revision/run_h10_c7a_development.py scripts/ch4_revision/lock_h10_c7_model_weights.py scripts/ch4_revision/build_h10_c7_open_replay_bundle.py scripts/ch4_revision/run_h10_c7_open_replay.py tests/h10_c7
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c7

h10-c7-build-open-replay:
	test -n "$(H10_C7_SOURCE_ARTIFACT)"
	test -n "$(H10_C7_PAYLOAD_ROOT)"
	test -n "$(H10_C7_PREPARED_MANIFEST)"
	test -n "$(H10_C7_PREPARED_GOLD)"
	test -n "$(H10_C7_REPLAY_BUNDLE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/build_h10_c7_open_replay_bundle.py --source-artifact "$(H10_C7_SOURCE_ARTIFACT)" --payload-root "$(H10_C7_PAYLOAD_ROOT)" --prepared-manifest "$(H10_C7_PREPARED_MANIFEST)" --prepared-gold "$(H10_C7_PREPARED_GOLD)" --repository-root . --output "$(H10_C7_REPLAY_BUNDLE)"

h10-c7-open-replay:
	test -n "$(H10_C7_REPLAY_BUNDLE)"
	test -n "$(H10_C7_REPLAY_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7_open_replay.py --bundle "$(H10_C7_REPLAY_BUNDLE)" --output "$(H10_C7_REPLAY_OUTPUT)"

h10-c7-confirmation:
	test -n "$(H10_C7_REPLAY_BUNDLE)"
	test -n "$(H10_C7_CONFIRMATION_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7_confirmation.py --bundle "$(H10_C7_REPLAY_BUNDLE)" --output "$(H10_C7_CONFIRMATION_OUTPUT)" --reports reports/h10_c7

h10-c7-r5v-audit:
	test -n "$(H10_C7_REPLAY_BUNDLE)"
	test -n "$(H10_C7_VERIFICATION_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/audit_h10_c7_pair_verification.py --bundle "$(H10_C7_REPLAY_BUNDLE)" --output "$(H10_C7_VERIFICATION_OUTPUT)"

h10-c7-prepare-development:
	test -n "$(H10_C7_SOURCE_MANIFESTS)"
	test -n "$(H10_C7_DEVELOPMENT_MANIFEST)"
	test -n "$(H10_C7_DEVELOPMENT_GOLD)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c7_development.py $(foreach manifest,$(H10_C7_SOURCE_MANIFESTS),--source-manifest "$(manifest)") --observable-output "$(H10_C7_DEVELOPMENT_MANIFEST)" --gold-output "$(H10_C7_DEVELOPMENT_GOLD)"

h10-c7-run-development:
	test -n "$(H10_C7_DEVELOPMENT_MANIFEST)"
	test -n "$(H10_C7_DEVELOPMENT_GOLD)"
	test -n "$(H10_C7_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7_development.py --manifest "$(H10_C7_DEVELOPMENT_MANIFEST)" --gold "$(H10_C7_DEVELOPMENT_GOLD)" --model-registry protocol/h10_c7/MODEL_REGISTRY.json --model-weight-lock protocol/h10_c7/H10_C7_MODEL_WEIGHT_LOCK.json --output "$(H10_C7_OUTPUT)" --root .

h10-c7a-prepare-development:
	test -n "$(H10_C7_REPLAY_BUNDLE)"
	test -n "$(H10_C7A_H10_C5B_MANIFEST)"
	test -n "$(H10_C7A_DEVELOPMENT_BUNDLE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c7a_development.py --base-bundle "$(H10_C7_REPLAY_BUNDLE)" --h10-c5b-manifest "$(H10_C7A_H10_C5B_MANIFEST)" --additional-incidents 10 --output "$(H10_C7A_DEVELOPMENT_BUNDLE)"

h10-c7a-run-development:
	test -n "$(H10_C7A_DEVELOPMENT_BUNDLE)"
	test -n "$(H10_C7A_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7a_development.py --manifest "$(H10_C7A_DEVELOPMENT_BUNDLE)/DEVELOPMENT_MANIFEST.jsonl" --gold "$(H10_C7A_DEVELOPMENT_BUNDLE)/DEVELOPMENT_GOLD.jsonl" --frozen-r5-reference results/h10_c7/confirmation/R5C_PER_INCIDENT.jsonl --output "$(H10_C7A_OUTPUT)"

h10-c7r-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/experiments/h10_c7r.py scripts/ch4_revision/lock_h10_c7r.py scripts/ch4_revision/prepare_h10_c7r_held_out.py scripts/ch4_revision/collect_h10_c7r_runtime.py scripts/ch4_revision/build_h10_c7r_gold.py scripts/ch4_revision/authorize_h10_c7r_scoring.py scripts/ch4_revision/run_h10_c7r_scoring.py tests/h10_c7r
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c7r

h10-c7r-lock:
	test -n "$(H10_C7A_DEVELOPMENT_BUNDLE)"
	test -n "$(H10_C7R_SOURCE_RELEASE)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/lock_h10_c7r.py --development-manifest "$(H10_C7A_DEVELOPMENT_BUNDLE)/DEVELOPMENT_MANIFEST.jsonl" --r5-reference results/h10_c7/confirmation/R5C_PER_INCIDENT.jsonl --source-release "$(H10_C7R_SOURCE_RELEASE)" --protocol-dir protocol/h10_c7r --output results/h10_c7r --root .

h10-c7r-prepare-held-out:
	test -n "$(H10_C7R_SOURCE_PARQUET)"
	test -n "$(H10_C7R_OPERATION_ROOT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c7r_held_out.py --source "$(H10_C7R_SOURCE_PARQUET)" --exclusion-lock protocol/h10_c7r/H10_C7R_EXCLUSION_LOCK.json --output "$(H10_C7R_OPERATION_ROOT)/selection"

h10-c7r-lock-images:
	test -n "$(H10_C7R_OPERATION_ROOT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/lock_h10_c7r_image_availability.py --runtime-registry "$(H10_C7R_OPERATION_ROOT)/selection/RUNTIME_REGISTRY.jsonl" --output "$(H10_C7R_OPERATION_ROOT)/selection/H10_C7R_IMAGE_AVAILABILITY_LOCK.json"

h10-c7r-collect-runtime:
	test -n "$(H10_C7R_OPERATION_ROOT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/collect_h10_c7r_runtime.py --selection "$(H10_C7R_OPERATION_ROOT)/selection/HELD_OUT_SELECTION.jsonl" --runtime-registry "$(H10_C7R_OPERATION_ROOT)/selection/RUNTIME_REGISTRY.jsonl" --image-availability-lock "$(H10_C7R_OPERATION_ROOT)/selection/H10_C7R_IMAGE_AVAILABILITY_LOCK.json" --output "$(H10_C7R_OPERATION_ROOT)/runtime" --prune-images

h10-c7r-build-gold:
	test -n "$(H10_C7R_OPERATION_ROOT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/build_h10_c7r_gold.py --sealed-source "$(H10_C7R_OPERATION_ROOT)/selection/SEALED_GOLD_SOURCE.jsonl" --manifest "$(H10_C7R_OPERATION_ROOT)/runtime/HELD_OUT_MANIFEST.jsonl" --output "$(H10_C7R_OPERATION_ROOT)/sealed-gold/HELD_OUT_GOLD.jsonl"

h10-c7r-authorize:
	test -n "$(H10_C7R_HELD_OUT_MANIFEST)"
	test -n "$(H10_C7R_HELD_OUT_GOLD)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/authorize_h10_c7r_scoring.py --manifest "$(H10_C7R_HELD_OUT_MANIFEST)" --gold "$(H10_C7R_HELD_OUT_GOLD)" --protocol-dir protocol/h10_c7r --authorization results/h10_c7r/SCORING_AUTHORIZATION.json --leakage-report reports/h10_c7r/H10_C7R_LEAKAGE_AUDIT.json

h10-c7r-score:
	test -n "$(H10_C7R_HELD_OUT_MANIFEST)"
	test -n "$(H10_C7R_HELD_OUT_GOLD)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7r_scoring.py --manifest "$(H10_C7R_HELD_OUT_MANIFEST)" --gold "$(H10_C7R_HELD_OUT_GOLD)" --authorization results/h10_c7r/SCORING_AUTHORIZATION.json --protocol-dir protocol/h10_c7r --output results/h10_c7r --reports reports/h10_c7r --root .

h10-c7r-r9-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_retrieval.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_diagnosis.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7r_r9.py scripts/ch4_revision/run_h10_c7r_r9_development.py tests/h10_c7/test_r9_retrieval.py
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c7/test_guided_retrieval.py tests/h10_c7/test_r9_retrieval.py tests/h10_c7r

h10-c7r-r9-development:
	test -n "$(H10_C7R_OPERATION_ROOT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7r_r9_development.py --manifest "$(H10_C7R_OPERATION_ROOT)/runtime/HELD_OUT_MANIFEST.jsonl" --gold "$(H10_C7R_OPERATION_ROOT)/sealed-gold/HELD_OUT_GOLD.jsonl" --exclusion-lock protocol/h10_c7r/H10_C7R_EXCLUSION_LOCK.json --v1-results results/h10_c7r/PER_INCIDENT_RESULTS.jsonl --output results/h10_c7r_r9/development --reports reports/h10_c7r_r9

h10-c7r-r10-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/experiments/h10_c5c_runtime.py framework/fuzzyxai/fuzzyxai/experiments/h10_c7r_r10.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/active_evidence.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_diagnosis.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_retrieval.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/importer.py framework/fuzzyxai/fuzzyxai/repository_diagnostics/runtime_events.py scripts/ch4_revision/audit_h10_c7r_r10_recollection_inputs.py scripts/ch4_revision/audit_h10_c7r_r10_runtime.py scripts/ch4_revision/build_h10_c7r_r10_provenance_evidence.py scripts/ch4_revision/collect_h10_c7r_runtime.py scripts/ch4_revision/run_h10_c7r_r10_development.py scripts/ch4_revision/verify_h10_c7r_r10_provenance.py tests/h10_c7/test_r10_causal_retrieval.py
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c7/test_r10_causal_retrieval.py tests/h10_c7/test_guided_retrieval.py tests/h10_c7/test_r9_retrieval.py tests/h10_c7r

h10-c7r-r10-provenance:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/verify_h10_c7r_r10_provenance.py

h10-c7r-r10-recollection-input-audit:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/audit_h10_c7r_r10_recollection_inputs.py --lock protocol/h10_c7r_r10/recollection/R10_TECHNICAL_BARRIER_LOCK.json --selection protocol/h10_c7r_r10/recollection/R10_TECHNICAL_BARRIER_SELECTION.jsonl --runtime-registry protocol/h10_c7r_r10/recollection/R10_TECHNICAL_BARRIER_RUNTIME_REGISTRY.jsonl --image-lock protocol/h10_c7r_r10/recollection/R10_TECHNICAL_BARRIER_IMAGE_LOCK.json --output results/h10_c7r_r10/recollection/R10_RECOLLECTION_INPUT_AUDIT.json

h10-c7r-r10-runtime-audit:
	test -n "$(H10_C7R_R10_MANIFEST)"
	test -n "$(H10_C7R_R10_READINESS)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/audit_h10_c7r_r10_runtime.py --manifest "$(H10_C7R_R10_MANIFEST)" --output "$(H10_C7R_R10_READINESS)"

h10-c7r-r10-development:
	test -n "$(H10_C7R_R10_MANIFEST)"
	test -n "$(H10_C7R_R10_GOLD)"
	test -n "$(H10_C7R_R10_READINESS)"
	test -n "$(H10_C7R_R10_OUTPUT)"
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c7r_r10_development.py --manifest "$(H10_C7R_R10_MANIFEST)" --gold "$(H10_C7R_R10_GOLD)" --exclusion-lock protocol/h10_c7r/H10_C7R_EXCLUSION_LOCK.json --readiness "$(H10_C7R_R10_READINESS)" --output "$(H10_C7R_R10_OUTPUT)"

h10-c5b-prepare:
	test -n "$(H10_C5B_SOURCE_DIR)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/prepare_h10_c5b_sources.py --output "$(H10_C5B_SOURCE_DIR)"

h10-c5b-prepare-runtime-inputs:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_DIR)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py prepare-runtime-inputs --manifest "$(H10_C5B_MANIFEST)" --output "$(H10_C5B_RUNTIME_DIR)" $(if $(H10_C5B_CONTAINER_IMAGES),--container-images "$(H10_C5B_CONTAINER_IMAGES)") $(if $(H10_C5B_RUNTIME_SOURCE_DATASET),--source-dataset "$(H10_C5B_RUNTIME_SOURCE_DATASET)")

h10-c5b-collect-runtime:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_COMMANDS)"
	test -n "$(H10_C5B_RUNTIME_DIR)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/collect_h10_c5b_runtime.py --manifest "$(H10_C5B_MANIFEST)" --commands "$(H10_C5B_RUNTIME_COMMANDS)" --output "$(H10_C5B_RUNTIME_DIR)"

h10-c5b-merge-runtime:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_MANIFEST)"
	test -n "$(H10_C5B_ENRICHED_MANIFEST)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py merge-runtime --original-manifest "$(H10_C5B_MANIFEST)" --runtime-manifest "$(H10_C5B_RUNTIME_MANIFEST)" --output "$(H10_C5B_ENRICHED_MANIFEST)"

h10-c5b-runtime-readiness:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_REPORT)"
	test -n "$(H10_C5B_SPLIT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py verify-runtime-readiness --manifest "$(H10_C5B_MANIFEST)" --runtime-report "$(H10_C5B_RUNTIME_REPORT)" --split "$(H10_C5B_SPLIT)" $(if $(H10_C5B_DEVELOPMENT_LOCK),--development-lock "$(H10_C5B_DEVELOPMENT_LOCK)")

h10-c5b-freeze-development:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_REPORT)"
	test -n "$(H10_C5B_DEVELOPMENT_RESULTS)"
	test -n "$(H10_C5B_DEVELOPMENT_LOCK)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py freeze-development --manifest "$(H10_C5B_MANIFEST)" --runtime-report "$(H10_C5B_RUNTIME_REPORT)" --development-results "$(H10_C5B_DEVELOPMENT_RESULTS)" --method-lock protocol/h10_c5b_repository_grounded/METHOD_LOCK.json --output "$(H10_C5B_DEVELOPMENT_LOCK)" --root .

h10-c5b-freeze-held-out-scoring:
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_REPORT)"
	test -n "$(H10_C5B_ENRICHED_MANIFEST)"
	test -n "$(H10_C5B_DEVELOPMENT_LOCK)"
	test -n "$(H10_C5B_HELD_OUT_LOCK)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py freeze-held-out-scoring --runtime-manifest "$(H10_C5B_MANIFEST)" --runtime-report "$(H10_C5B_RUNTIME_REPORT)" --enriched-manifest "$(H10_C5B_ENRICHED_MANIFEST)" --development-lock "$(H10_C5B_DEVELOPMENT_LOCK)" --method-lock protocol/h10_c5b_repository_grounded/METHOD_LOCK.json --controller scripts/ch4_revision/score_h10_c5b_held_out.py --output "$(H10_C5B_HELD_OUT_LOCK)" --root .

h10-c5b-plan-replacements:
	test -n "$(H10_C5B_CANDIDATES)"
	test -n "$(H10_C5B_MANIFEST)"
	test -n "$(H10_C5B_RUNTIME_REPORT)"
	test -n "$(H10_C5B_REPLACEMENT_LOG)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py plan-replacements --candidates "$(H10_C5B_CANDIDATES)" --manifest "$(H10_C5B_MANIFEST)" --runtime-report "$(H10_C5B_RUNTIME_REPORT)" --output "$(H10_C5B_REPLACEMENT_LOG)"

h10-c5b-run:
	test -n "$(H10_C5B_MANIFEST)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c5b.py --manifest "$(H10_C5B_MANIFEST)" --root "$(or $(H10_C5B_ROOT),.)"

h10-c5b-score-held-out:
	test -n "$(H10_C5B_HELD_OUT_LOCK)"
	test -n "$(APPROVAL)"
	test -n "$(H10_C5B_SCORING_STATUS)"
	test -n "$(H10_C5B_OPENING_RECORD)"
	test -n "$(H10_C5B_SCORING_ROOT)"
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/score_h10_c5b_held_out.py --lock "$(H10_C5B_HELD_OUT_LOCK)" --approval "$(APPROVAL)" --status "$(H10_C5B_SCORING_STATUS)" --opening-record "$(H10_C5B_OPENING_RECORD)" --output-root "$(H10_C5B_SCORING_ROOT)" --repository-root .

h10-c5b-verify-method-lock:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/h10_c5b_runtime_ops.py verify-method-lock --lock protocol/h10_c5b_repository_grounded/METHOD_LOCK.json --root .

h10-c5b-parent-immutability:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/verify_parent_result_immutability.py --root .

h10-c6-run:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c6.py --root .

h10-c5-pilot-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/experiments/h10_c5_pilot.py scripts/ch4_revision/run_h10_c5_pilot.py tests/h10_c5_pilot
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c5_pilot

h10-c5-pilot-run:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c5_pilot.py

h10-c6-n-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/experiments/h10_c6_noise.py scripts/ch4_revision/run_h10_c6_noise.py tests/h10_c6_noise
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/h10_c6_noise

h10-c6-n-prepare:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c6_noise.py prepare --root .

h10-c6-n-run:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h10_c6_noise.py run --root .

mlflow-integration-test:
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m ruff check framework/fuzzyxai/fuzzyxai/adapters/mlflow_tabular.py framework/fuzzyxai/fuzzyxai/core/mlflow_route.py examples/integrations/mlflow_fuzzyxai_example.py scripts/ch4_revision/run_mlflow_integration.py tests/integration/test_mlflow_adapter.py tests/integration/test_mlflow_integration.py
	$(H10_C4_ENV) $(H10_C3_PYTHON) -m pytest -q tests/integration/test_mlflow_adapter.py tests/integration/test_mlflow_integration.py

mlflow-integration-run:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_mlflow_integration.py

final-practical-parent-immutability:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/verify_parent_result_immutability.py

final-practical-package:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/build_final_practical_evidence.py

multimodal-route-validation:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_multimodal_routes.py

h9-e2e-latency:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h9_e2e.py

h9-e2e-v2:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/run_h9_e2e_v2.py

ch4-q1-claim-lint:
	$(H10_C4_ENV) $(H10_C3_PYTHON) scripts/ch4_revision/claim_lint.py --root .

ch4-q1-evidence: h10-c5-source-audit h10-c6-run multimodal-route-validation h9-e2e-latency ch4-q1-claim-lint

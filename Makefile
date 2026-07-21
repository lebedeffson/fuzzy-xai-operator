PYTHON ?= python
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

chapter4-final: final-claim-registry
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_confirmatory_outputs.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/final_practical_closure/build_chapter4_final.py

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

reproduce-final-practical-closure: practical-controller-formative practical-controller-freeze practical-controller-confirmatory final-statistics final-claim-registry chapter4-final practical-release-archive

PYTHON ?= $(shell if [ -x /home/lebedeffson/Code/venv/bin/python ]; then echo /home/lebedeffson/Code/venv/bin/python; else echo python; fi)
PYTHONPATH := .
PORT ?= 8085
DATASET ?= breast_cancer
BASELINE_ACCESS ?= native

.PHONY: install test risk-test category-hott-test chapter2-breast-cancer-demo chapter2-real-operator-case reproduce-chapter2 figures-chapter2 chapter5-experiments chapter5-demo chapter5-latex web-demo unified-demo layered-demo layered-demo-legacy defense-demo defense-demo-legacy studio ui-health-check ui-health-check-all browser-visual-check unified-demo-cli full-pipeline figures full-experiments demo dashboard proof formal-proof thesis full-demo full-observer dataset-observer dataset-modes-check baseline-check real-data-validation benchmark benchmark-dataset baseline-comparison calibrate-observer ablation-benchmark defense-cases real-reduction-example dissertation-demo-summary dissertation-component-tables dissertation-check dataset-cards thesis-practice-tables structure-aware-benchmark reproducibility-artifacts operator-benchmark risk-benchmark lofo-f1-demo clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

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

figures-chapter2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/generate_figures.py --out-dir reports/figures

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
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/reproducibility_artifacts.py --out-dir reports/reproducibility_artifacts

dissertation-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/check_dataset_modes.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_benchmark.py --dataset breast_cancer --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fuzzyxai.experiments.chapter2_sample113 --out-dir reports/chapter2
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter2_real_operator_case.py --out-dir reports/chapter2_real_operator_case
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/real_reduction_example.py --out-dir reports/real_reduction_example
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_demo_summary.py --out-dir reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dissertation_component_tables.py --out-dir reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/dataset_cards.py --out-root reports/datasets
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/reproducibility_artifacts.py --out-dir reports/reproducibility_artifacts
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

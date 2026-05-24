PYTHON ?= $(shell if [ -x /home/lebedeffson/Code/venv/bin/python ]; then echo /home/lebedeffson/Code/venv/bin/python; else echo python; fi)
PYTHONPATH := .
PORT ?= 8085

.PHONY: install test risk-test category-hott-test chapter2-breast-cancer-demo chapter5-experiments chapter5-demo chapter5-latex full-pipeline figures full-experiments demo dashboard proof formal-proof thesis full-demo full-observer dataset-observer benchmark operator-benchmark risk-benchmark lofo-f1-demo clean

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

chapter5-experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/chapter5_experiments.py --n-per-scenario 1000 --timing-n 1000 --out-dir reports/chapter5

chapter5-demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m experiments.chapter5_demo

chapter5-latex:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/export_chapter5_latex.py

full-pipeline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/full_pipeline_demo.py --out-dir reports/full_pipeline

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/generate_figures.py --out-dir reports/figures

full-experiments: chapter5-experiments chapter2-breast-cancer-demo full-pipeline figures chapter5-latex
	@echo "All experiments completed. Reports are in reports/."

demo:
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

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/breast_cancer_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/operator_comparison_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/risk_aware_observer_benchmark.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/lofo_f1_rule_pruning_demo.py

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

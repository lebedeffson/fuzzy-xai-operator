# Baseline check

```
........................................................................ [ 51%]
....................................................................     [100%]
=============================== warnings summary ===============================
tests/test_chapter5_web_demo_smoke.py::test_chapter5_web_demo_backend_smoke
tests/test_chapter5_web_demo_smoke.py::test_chapter5_web_demo_backend_smoke
tests/test_real_reduction_example.py::test_real_reduction_example_generates_outputs
tests/test_real_reduction_example.py::test_real_reduction_example_generates_outputs
tests/test_real_reduction_example.py::test_real_reduction_example_generates_outputs
  /home/lebedeffson/Code/venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
    warnings.warn(

tests/test_real_reduction_example.py::test_real_reduction_example_generates_outputs
  /home/lebedeffson/Code/venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2749: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
140 passed, 6 warnings in 10.98s

dataset_mode	status	rows	domain	validates	details
breast_cancer	READY	569	medical	risk observer baseline	sklearn.datasets
diabetes_binary	READY	442	medical	borderline uncertainty	sklearn.datasets
wine_risk	READY	178	tabular	transferability	sklearn.datasets
synthetic_ruptures	READY	900	diagnostic	controlled ruptures	fuzzyxai.synthetic
registry_programs	MISSING	-	text-tabular	registry records	data/cit_registry/registry_programs.csv
registry_mosmed_doctor_analysis	MISSING	-	medical audit	doctor/model audit	data/cit_registry/registry_mosmed_doctor_analysis.csv
registry_steel_ir	MISSING	-	industrial CV	industrial transferability	data/cit_registry/registry_steel_ir.csv

{
  "dataset": "breast_cancer",
  "status": "READY",
  "n": 569,
  "domain": "medical",
  "model_accuracy": 0.972027972027972,
  "model_roc_auc": 0.9949685534591195,
  "observer_action_accuracy": 0.5594405594405595,
  "mean_I_pre": 0.7935691880178595,
  "mean_rho": 0.2076536833161616,
  "rupture_rate": 0.0,
  "action_distribution": {
    "lower_confidence": 36,
    "accept": 90,
    "request_more_data": 17
  },
  "selected_representation_distribution": {
    "FML-audit": 143
  },
  "notes": "Prototype measurements per object; no I/O timing."
}
```

# FuzzyXAI ML Vertical v1 acceptance

Status: `FUZZYXAI_ML_VERTICAL_V1_IMPLEMENTED`

The vertical uses the registered sklearn model, real SHAP values, fuzzy uncertainty representations, diagnostic contracts, repair/recertification, deterministic audience views, and MLflow artifacts. It is a reproducible software demonstration and makes no clinical or human-time claim.

```json
{
  "scenario_count": 10,
  "route_executability_rate": 1.0,
  "valid_route_rate": 0.4,
  "false_certification_rate": 0,
  "representation_selection_accuracy": 1,
  "observer_action_accuracy": 1,
  "repair_plan_success_rate": 1,
  "recertification_success_rate": 1,
  "provenance_completeness_rate": 0.8,
  "reduction_fidelity_rate": 0.7,
  "audience_consistency_rate": 1,
  "artifact_integrity_rate": 1,
  "registered_violation_detection_rate": 1,
  "mean_model_ms": 1.2764070999764954,
  "mean_shap_ms": 0.9925691999796982,
  "mean_fuzzyxai_ms": 7.763796200106299,
  "mean_total_ms": 10.032772500062492,
  "mean_artifact_size_bytes": 89556.9,
  "representation_counts": {
    "F0": 6,
    "F_int": 2,
    "NAS": 1,
    "F_ML": 1
  },
  "action_counts": {
    "ACCEPT": 3,
    "WARN": 2,
    "REQUEST_DATA": 1,
    "REVIEW": 2,
    "BLOCK": 2
  },
  "mlflow_logged_runs": 10
}
```

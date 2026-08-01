from __future__ import annotations

from fuzzyxai.adapters import get_adapter, list_adapters


def test_mlflow_adapter_requires_registered_provenance() -> None:
    adapter = get_adapter("mlflow_tabular")
    validation = adapter.validate_payload(
        {
            "scenario_id": "mlflow_tabular_classification",
        }
    )
    assert not validation.valid
    assert "missing fields" in validation.errors[0]


def test_mlflow_adapter_is_publicly_registered() -> None:
    adapter = get_adapter("mlflow_tabular")
    assert adapter.describe()["adapter_id"] == "mlflow_tabular"
    assert "mlflow_tabular" in {
        item["adapter_id"]
        for item in list_adapters()
    }

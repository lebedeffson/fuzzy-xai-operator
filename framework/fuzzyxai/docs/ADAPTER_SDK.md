# Adapter SDK

Adapters convert external model payloads into `AdaptedInput`.

Required methods:

```python
class BaseAdapter:
    adapter_id: str
    task_type: str
    supported_payload_schema: str

    def validate_payload(self, payload: dict) -> ValidationResult: ...
    def to_adapted_input(self, payload: dict) -> AdaptedInput: ...
    def describe(self) -> dict: ...
```

Registry usage:

```python
from fuzzyxai.adapters import get_adapter

adapter = get_adapter("tabular_classification")
```

External repositories should publish payloads that pass one of the schemas in
`fuzzyxai/schemas/`.

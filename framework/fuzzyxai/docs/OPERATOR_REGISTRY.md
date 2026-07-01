# Operator Registry

The operator registry describes the computational route:

```python
from fuzzyxai.operators import list_operators

for operator in list_operators():
    print(operator["operator_id"], operator["formula_text"])
```

Registered operators:

- `input_artifact`
- `explanation_object`
- `representation`
- `alignment`
- `reduction`
- `risk`
- `diagnostics`
- `action`
- `proof`

Each entry includes input/output contracts, formula text, required components and
produced values. The registry is used by CLI checks, documentation and verifier
traceability audits.

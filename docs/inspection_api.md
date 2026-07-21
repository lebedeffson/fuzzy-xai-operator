# Inspection API

`result.inspect()` returns a typed `InspectionResult` with `summary()`, `evidence()`, `limitations()`, `provenance()`, `audit()`, and `visualize()`.

Supported selectors:

```python
result.inspect("claim:C004")
result.inspect("rule:R31")
result.inspect("concept:1")
result.inspect("object:85")
result.inspect("evidence:data:85")
result.inspect("diagnostic:0")
result.inspect("action")
```

Unknown targets raise `KeyError`; malformed selectors raise `ValueError`. Inspection never creates additional evidence.

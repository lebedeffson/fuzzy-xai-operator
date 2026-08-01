# FuzzyXAI Python Package

This directory contains the installable FuzzyXAI framework. The canonical
public API is:

```python
from fuzzyxai import FuzzyXAI

result = FuzzyXAI.wrap(model, adapter="auto", task="auto").explain(inputs)
```

Use the repository-level [README](../../README.md) for installation,
architecture, diagnostics, validation evidence, and release boundaries.
Package-specific references are available in [`docs/`](docs/).

The canonical visualization namespace is `fuzzyxai.visualization`. The
`fuzzyxai.visual` and `fuzzyxai.viz` modules are compatibility shims only.

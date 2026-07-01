# FuzzyXAI CLI

Install:

```bash
pip install -e framework/fuzzyxai
```

Commands:

```bash
fuzzyxai run --payload payload.json --adapter tabular_classification --out out/
fuzzyxai verify --route out/route.json --proof out/proof_trace.json
fuzzyxai render --route out/route.json --out dashboard.png
fuzzyxai package --route out/route.json --out audit_package.zip
fuzzyxai validate --payload payload.json --schema classification
fuzzyxai list-adapters
fuzzyxai list-operators
```

The CLI uses the installed `fuzzyxai` package and does not import
`applications/scenarios` or the DubnaXAI site.

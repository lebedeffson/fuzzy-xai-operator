# Chapter 4 candidate evidence

Build the computed package with:

```bash
python experiments/model_universality/run_benchmark.py
python experiments/model_universality/runtime_validation.py --library sklearn --output /tmp/adapter-reports
python scripts/merge_model_validation_reports.py --reports-dir /tmp/adapter-reports
python scripts/verify_model_universality.py
python scripts/verify_explanation_quality.py
python scripts/build_external_validation_package.py
python scripts/build_chapter4_final_candidate.py
python scripts/verify_chapter4_final_candidate.py
```

The committed package uses the merged GitHub Actions artifact rather than a workstation-only optional-runtime
claim. It contains 34 deterministic core configurations and six optional runtime integrations, each measured on
Python 3.11 and Python 3.12. The package combines checkpoint evidence, native rule ablation, similar-case
provenance, adapter conformance, prediction parity, API checks, explanation-quality checks, and the state of
external gates. It remains a `candidate` while the independent pilot or domain review is incomplete.

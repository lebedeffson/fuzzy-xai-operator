# Chapter 4 candidate evidence

Build the computed package with:

```bash
python experiments/model_universality/run_benchmark.py
python scripts/build_external_validation_package.py
python scripts/build_chapter4_final_candidate.py
python scripts/verify_chapter4_final_candidate.py
```

The package combines measured checkpoint evidence, native rule ablation, similar-case provenance, model-adapter
conformance, prediction parity, human explanations, and the state of external gates. It is a `candidate` while the
pilot or domain review is incomplete. The report must preserve that distinction when Chapter 4 is written.

# H10-C7 reproduction

```bash
make h10-c7-test H10_C3_PYTHON=python
```

The real development tournament additionally requires:

```bash
make h10-c7-run-development \
  H10_C3_PYTHON=python \
  H10_C7_DEVELOPMENT_MANIFEST=/secure/observable-development.jsonl \
  H10_C7_DEVELOPMENT_GOLD=/secure/development-gold.jsonl \
  H10_C7_OUTPUT=results/h10_c7
```

The model registry must contain locally verified weight SHA256 values. The
runner does not download models and does not create held-out data. A method
lock is written only if every development gate passes.

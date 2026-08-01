# Reproduction

The protocol lock must be committed before scoring.

```bash
PYTHONPATH=framework/fuzzyxai:. python scripts/ml_pipeline_v2_comparative/run_evaluation.py
make ml-pipeline-v2-comparative-test
```

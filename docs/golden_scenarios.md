# Golden explanation scenarios

The deterministic builder is:

```bash
PYTHONPATH=framework/fuzzyxai python scripts/build_explanation_experience_evidence.py
```

It creates:

- object 85: twelve checkpoints, forgetting at epoch 16, global/subgroup divergence, R31 ablation;
- ANFIS: native rules and class knowledge;
- black-box callable: honest missing internal channels;
- sklearn linear: native coefficient contributions and surrogate rule-like statements;
- decision tree: native paths;
- medical research fixture: saved images, masks, measured IoU, embedding metric, and two counterexamples.

`release_evidence/explanation_experience/manifest_sha256.json` is rebuilt from generated artifacts. Golden JSON and figures must not be edited manually.

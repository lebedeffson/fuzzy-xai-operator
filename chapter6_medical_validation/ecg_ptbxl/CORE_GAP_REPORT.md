# Frozen-core boundary for ECG cross-XAI alignment

The Chapter 6 ECG experiment preserves real 12x1000 Integrated Gradients and
real 12x20 temporal-occlusion evidence. Both are transformed to the same signed
12x20 L1-normalized grid and their distance is reported as a diagnostic.

The frozen P19 public runtime currently creates its canonical `SystemEvidence`
from the model probability interface and one ExplainPlan-owned target
interface. It does not accept two experiment-side spatial/temporal
`ExplanationObject` instances through `explain_one()` for a second canonical
Gamma route, nor a caller-registered 12x1000 -> 12x20 Pi/iota reduction.

Therefore:

- the public SystemEvidence Gamma is the real registered
  `ecg_binary_probability -> technical_risk` alignment;
- IG/occlusion normalized-L1 distance is explicitly diagnostic and is not
  renamed Gamma;
- temporal reduction has status `not_applied` and `w_Delta=0`;
- no frozen framework code is changed to make the Chapter 6 table look full.

Supporting arbitrary typed XAI-to-XAI alignment/reduction through the same
public call is a post-P19 core extension, not part of this validation pass.

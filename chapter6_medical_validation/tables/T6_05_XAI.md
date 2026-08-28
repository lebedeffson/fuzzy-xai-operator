| domain | method | representation | faithfulness | spatial_diagnostic | limitations |
| --- | --- | --- | --- | --- | --- |
| ECG | Integrated Gradients | 12×1000 signed tensor, fixed target logit | top-10% masking vs random in per-case xai_diagnostics.json | not applicable | local sensitivity; not physiology causality |
| ECG | temporal occlusion | 12×20 signed windows | top-10% masking vs random | 12×20 common-grid diagnostic, not Gamma | experiment-side secondary source |
| brain_v2_confirmatory | Grad-CAM + IG | 299×299 map / full tensor | not supplied | HPF overlap only on positive patches; not Gamma | single atlas, no causal claim |
| eyes | Grad-CAM + IG scaffold | not executed | MISSING_DATA | MISSING_DATA | official IDRiD access required |

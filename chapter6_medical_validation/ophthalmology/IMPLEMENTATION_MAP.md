# Frozen-framework implementation map

1. **PyTorch classifier.** `fuzzyxai.adapters.optional_v2.TorchAdapter` wraps a
   `torch.nn.Module`, applies an optional `input_transform`, emits softmax
   probabilities and native autograd IG evidence.
2. **Raw image.** `FuzzyXAI.wrap(...).explain_one(numeric_input,
   raw_object=image)` sends the tensor-compatible numeric input to the adapter
   and preserves the raw 2D/3D image for first-class image evidence.
3. **Attribution evidence.** TorchAdapter places the full IG tensor in its
   typed local evidence. Runtime converts known channels through
   `build_attribution_map` and preserves tensor, target, baseline and
   completeness.
4. **External Grad-CAM.** It can be computed before `explain_one()`, converted
   by public `build_attribution_map`, and supplied as typed
   `ExplanationEvidence.attribution_maps` via `additional_evidence`. No core
   patch is needed.
5. **Multiclass system source.** Frozen runtime derives generic
   `SystemSourceEvidence` from `ModelPrediction.probabilities` and class labels.
   The technical system coordinate is a registered class probability; the full
   vector and derived referable-DR probability remain separately disclosed.
6. **Preprocessing provenance.** Relative sample ID, raw SHA256, config SHA256,
   deterministic operation trace and output hash are supplied as typed
   `DataEvidence` plus `ObservationContext.run_parameters`.
7. **Image quality.** Technical blur/exposure/FOV measurements are a second
   typed `DataEvidence` record. They are not clinical image-quality claims.
8. **Reduction.** P19 can reduce its declared uncertainty interval
   `F_int→F0`; it cannot represent full-image attribution Pi/iota from an
   experiment-side adapter. The eye plan therefore declares image-attribution
   reduction `not_applied`; no manual Delta is computed.
9. **System route.** Yes: class probabilities + registered factual trace +
   matching ExplainPlan transform allow public runtime to produce T_ij, Gamma,
   entropy U_model, U_trace, u_M, E_pre, I_pre, strict rho and action. Native
   rules are `not_applicable` with zero eta/weight.
10. **Result fields.** `ModelExplanationResult` exposes prediction, typed
    evidence, `system`, audit, reader/audit reports, quality/capability reports,
    inspect and ExplanationGraph. Scientific system numbers are read from this
    result only.

## Disclosed gaps

- Grad-CAM is accepted as a typed external native-model evidence channel but
  frozen runtime does not automatically use Grad-CAM-vs-IG spatial comparison
  as canonical Gamma. Calling IoU or heatmap agreement Gamma would be invalid.
- Full spatial attribution reduction needs a future separately approved core
  contract; this experiment remains `not_applied` for that Delta.
- Current local environment imports `torch` but its installed `torchvision`
  fails at `torchvision::nms`. Model construction is guarded with an actionable
  error and must be repaired in the environment before training.

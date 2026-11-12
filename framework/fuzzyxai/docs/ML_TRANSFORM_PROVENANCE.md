# Numeric preprocessing provenance (read-only FuzzyXAI observer)

Purpose: connect raw training/evaluation rows to the outputs of one
explicitly invoked numeric transformer and to subsequent model fit and
inference inputs. The observer itself NEVER invokes fit, transform,
predict, predict_proba or repair.

## Typical caller-initiated StandardScaler pipeline

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from fuzzyxai.fit_provenance import FitRecorder
    from fuzzyxai.transform_provenance import TransformRecorder
    from fuzzyxai.output_provenance import OutputRecorder
    from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run

    train_ids = tuple(train_row_ids)
    eval_ids = tuple(evaluation_row_ids)
    fit = FitRecorder(target_hmac_key=secret_key)
    scaler = fit.fit(
        StandardScaler(), X_train,
        role="preprocessor", row_ids=train_ids,
    )
    transforms = TransformRecorder()
    Xt = transforms.transform(
        scaler, X_train, split="train", row_ids=train_ids,
    )
    Xe = transforms.transform(
        scaler, X_test, split="evaluation", row_ids=eval_ids,
    )
    model = fit.fit(
        LogisticRegression(max_iter=800), Xt, y_train,
        role="model", row_ids=train_ids, feature_space="transformed",
    )
    outputs = OutputRecorder()
    predictions = outputs.predict(
        model, Xe, row_ids=eval_ids, feature_space="transformed",
    )
    probabilities = outputs.predict_proba(
        model, Xe, row_ids=eval_ids, feature_space="transformed",
    )
    evidence = MLRunEvidence(
        task="classification", model_family=type(model).__name__,
        train_features=X_train, evaluation_features=X_test,
        train_targets=y_train, evaluation_targets=y_test,
        train_ids=train_ids, evaluation_ids=eval_ids,
        model_fit_ids=train_ids, preprocessor_fit_ids=train_ids,
        preprocessor_applied=True,
        predictions=predictions, probabilities=probabilities,
        probability_classes=tuple(str(c) for c in model.classes_),
        fit_records=[item.to_dict() for item in fit.records],
        transform_records=[item.to_dict() for item in transforms.records],
        output_records=[item.to_dict() for item in outputs.records],
    )
    result = observe_ml_run(evidence, target_hmac_key=secret_key)

The secret target HMAC key is supplied out of band. Never put it into
the evidence JSON, repo or logs. The read-only command-line report can
inspect saved evidence without accessing the original model, but cannot
verify target HMACs unless the secret is explicitly available to the
Python API; it must return UNKNOWN instead of claiming success.

## Report fields and boundaries

- transform_train_input: ordered input ID and numeric-content match
  to the declared raw training partition.
- transform_evaluation_input: same for the evaluation partition.
- transform_identity: matching caller-declared transformer type and tag
  across train/evaluation. Actual object identity is NOT authenticated.
- transform_preprocessor_fit_link: witnessed preprocessor fit input/type
  agrees with the training transform source.
- transform_model_fit_link: witnessed transformed model fit inputs agree
  with the training transform output.
- transform_predict_link and transform_predict_proba_link: witnessed
  transformed prediction inputs agree with evaluation transform output.
  Non-classification probability checks are NOT_APPLICABLE.

Earlier raw-only fit_trace_model_content and output_trace_predict_inputs
may remain UNKNOWN on transformed pipelines. They are intentionally
not silently promoted to success by unrelated evidence; the explicit
new transform linkage checks represent the extra confidence.

This deliberately covers ONE dense numeric preprocessor, a single
unambiguous train batch and evaluation batch, and their recorded model
calls. No multi-stage chained transforms, sparse/missing values,
text/image/LLM/RAG types, partial/streaming batches or unsupported
outputs are certified. Missing or ambiguous traces yield UNKNOWN.

All digests and transformer tags are local claims. Their agreement does
not prove trustworthy data origin, model/transformer object identity,
absence of uninstrumented calls, correct source labels, or model safety.
An adversary controlling the recorder and both declared sources can
forge a consistent bundle. The auditing code cannot repair anything.

## Developmental experiment

Frozen before measurement:
experiments/ml_transform_chain_protocol_v1.json.
Reproduce with:
python -m experiments.ml_transform_chain_benchmark

Four fitted sklearn pipelines (Breast Cancer/LogisticRegression,
Wine/LogisticRegression, Diabetes/Ridge, Iris/KMeans); six controlled
scenarios each. StandardScaler always receives an explicit real fit
invocation, and its transform is explicitly requested by the caller.
Benchmark compares a metadata-only baseline with a strong direct
comparator that sees the same fit, transform and inference arrays.
These are 24 correlated developer-constructed scenarios and are
NOT independent real-world fault-sensitivity estimates. Equal-access
direct comparison can discover the same injected discrepancies.

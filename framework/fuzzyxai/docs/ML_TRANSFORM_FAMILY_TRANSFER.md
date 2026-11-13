# Cross-transform-family transfer: developmental validation

Date: 2026-11-13. Draft PR #10. This report concerns the unchanged, read-only FuzzyXAI transform-chain observer. No automatic model training, inference, transformation, or repair is performed by the observer.

## Frozen protocol and reproducibility

Protocol: experiments/ml_transform_family_transfer_protocol_v1.json.
SHA256: ae6269821ac371225ef973c122e3cdcddf5fffb6a6dc6581c4e11b42dacd3859.
The protocol was committed before benchmark implementation.
Runner: python -m experiments.ml_transform_family_transfer_benchmark.
Regression: tests/test_ml_transform_family_transfer.py.
Runtime source SHA: 2075b876e08a675a8f596b1ba708dfeb0f276817.
GitHub Actions 43378583969 succeeded: Python 3.11.18 and 3.12.16, **76 pytest passed and Ruff passed per interpreter**. Both real JSON artifacts were downloaded and compared: 14014524457 (3.11) and 14015018769 (3.12). Their protocol hashes, 64 case records, outcomes and limitations match. NumPy versions differed (2.4.6 versus 2.5.4); sklearn 1.9.1 was used for both.

## Eight real sklearn pipelines, six source datasets

| Dataset | Transformer | Model | Raw -> transformed width |
| --- | --- | --- | --- |
| Digits | PCA | KNeighborsClassifier | 64 -> 12 |
| Wine | RobustScaler | SVC | 13 -> 13 |
| Breast Cancer | QuantileTransformer | LogisticRegression | 30 -> 30 |
| Diabetes | SplineTransformer | Ridge | 10 -> 40 |
| Linnerud (first regression target) | MinMaxScaler | Ridge | 3 -> 3 |
| Iris | PowerTransformer | RandomForestClassifier | 4 -> 4 |
| Digits | MaxAbsScaler | GaussianNB | 64 -> 64 |
| Iris | StandardScaler | KMeans | 4 -> 4 |

The Linnerud evaluation uses one regression response because general regression metrics currently support a scalar response. An independent training-target HMAC API accepts 2D labels, but this does not prove multi-output evaluation metrics.

## Fixed scenario results

| Scenario | N | Metadata-only alerts | FuzzyXAI transform-link alerts | Equal-evidence direct comparator |
| --- | ---: | ---: | ---: | ---: |
| Clean | 8 | 0 | 0 | 0 |
| Coherent shifted evaluation input | 8 | 0 | 0 | 0 |
| Coherent full retraining | 8 | 0 | 0 | 0 |
| Different model-fit transformed values | 8 | 0 | 8 | 8 |
| Different inference transformed values | 8 | 0 | 8 | 8 |
| Different declared raw evaluation values | 8 | 0 | 8 | 8 |
| Preprocessor fit on evaluation rows | 8 | 8 | 8 | 8 |
| No transform witness | 8 | 0 | 0 (8 UNKNOWN) | 0 |

Total: 64 *correlated, developer-constructed* cases; 24 valid, 32 known invalid, 8 with missing witness. In the **predeclared transform/split check subset**, FuzzyXAI detected 32/32 known violations, raised 0/24 alerts in valid cases, and returned UNKNOWN for all 8 missing-witness cases. The metadata-only baseline found 8/32, and an equally privileged direct array comparator found 32/32. These numbers do **not** describe the false alarm rate of unrelated model-output quality checks.

## Limits

These are manually instrumented pipelines and deliberately introduced faults, not naturally occurring incidents or independently blinded data. The observer verifies caller-provided consistency, not the authenticity of weights, true labels, original source, or completeness of execution tracing. A forged coherent trace is possible. The test does not establish universal model accuracy, causal correctness, privacy protection, support for arbitrary multi-stage / sparse / text / CV / LLM / RAG pipelines, or superiority over tools with equal access to the same arrays.

No locked scored dissertation artifacts or four manuscripts were changed. The PR remains draft and stacked; this scoped successful CI is not a full-repository release certificate.

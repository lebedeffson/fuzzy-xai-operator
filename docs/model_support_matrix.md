# Model support and claim boundary

The generated matrix is `release_evidence/model_universality/support_matrix.csv`. Its canonical summary and
checksums are in the same directory.

Current core benchmark scope:

- 24 classification configurations;
- 10 regression configurations;
- sklearn linear, tree, ensemble, SVM, KNN, Naive Bayes, pipeline, and generic model contracts;
- measured prediction parity, adapter conformance, graph validation, and serialized human explanation.

Optional boosting and neural runtimes have separate CI jobs. Their code presence is not a verification claim.
Only a row with `status=verified` or a completed runtime-specific CI job is eligible for release documentation.

Universality means one API with family-specific evidence and explicit limitations. It does not mean an identical
explanation for every model, and it does not imply universal support for arbitrary preprocessing, custom tensors,
tokenization, image segmentation, forecasting horizons, or unavailable training checkpoints.

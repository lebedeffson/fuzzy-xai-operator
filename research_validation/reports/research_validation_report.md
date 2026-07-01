# FuzzyXAI Research Validation Report

## Goal
Проверить переносимость FuzzyXAI на разных классах задач, моделях и типах ограничений. Проверка показывает работу операторного слоя, а не заявляет промышленную, клиническую или биометрическую применимость.

## Experiment Matrix
Всего экспериментов: 20. Матрица включает чистые входы, top-k редукцию, missing features, noise, confidence boundary, explanation conflict, wide interval и image/signal quality limits.

## Models and Task Classes
Классы задач: image_like_classification, signal_quality, tabular_classification, tabular_regression.
Модели: DecisionTreeClassifier, GradientBoostingClassifier, GradientBoostingRegressor, KNeighborsClassifier, LinearRegression, LogisticRegression, MLPClassifier, MLPRegressor, RandomForestRegressor, SVC(probability=True), SVR, threshold_model.

## Representation Class Coverage
F0: 3, F_ML: 9, F_int: 4, NAS: 4

## Operator Behavior
Во всех экспериментах FuzzyXAI сформировал gamma, delta, rho и выбрал действие через operator route. Gamma реагирует на неопределённость, качество, конфликт и интервальную ширину; delta фиксирует потери top-k объяснения; rho агрегирует доминирующий компонент риска.

## Action Distribution
accept: 4, audit: 3, defer_to_human: 2, lower_confidence: 11

## Diagnostic Distribution
D_external_image_uncertainty: 1, D_external_regression_explanation_loss: 1, D_external_regression_uncertainty: 3, D_external_tabular_ok: 4, D_external_tabular_quality_limit: 2, D_external_tabular_reduction_loss: 1, D_external_tabular_uncertainty: 3, D_image_explanation_reduction: 1, D_image_quality_limit: 1, D_rule_attribution_conflict: 1, D_signal_missing_fragments: 1, D_signal_noise_limit: 1

## Traceability Verification
verifier passed: 20 / 20.
traceability passed: 20 / 20.

## Key Findings
- FuzzyXAI produced non-zero operator values across several model families.
- Dominant risk components: delta=6, gamma=14.
- Representation classes activated: F0, F_ML, F_int, NAS.
- All generated routes passed verifier and traceability checks.

## Limitations
- This suite is deterministic payload-level research validation, not a claim of production performance.
- Signal and image-like experiments are methodological controls and do not imply clinical or biometric applicability.
- External repository payload contracts still require pinned live adapters.

## Files and Reproducibility
- research_validation_results.csv
- action_distribution.csv
- diagnostic_distribution.csv
- representation_class_coverage.csv
- risk_component_summary.csv
- manifest.json with sha256 checksums; manifest self-hash is excluded by policy
- outputs/<experiment_id>/

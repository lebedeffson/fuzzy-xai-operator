# Глава 6 — офтальмология: PAPILA

## Данные и frozen protocol

Офтальмологическая empirical line выполнена на официальном PAPILA v2: 244 пациента, 488 RGB fundus images; healthy=333, glaucoma=87, suspect=68. Primary binary protocol исключает целиком suspect-associated patients: clean binary cohort=210 patients, auxiliary suspect cohort=34 patients. ResNet50, patient-level five-fold split (seed 2026) и canonical outer fold 5 / seed 2026 заморожены до native-XAI анализа. Fixed-seed CV: AUROC=0.6874±0.0637, balanced accuracy=0.5990±0.0718. Это умеренный, а не скрытый «красивый» результат; FuzzyXAI не улучшает classifier performance.

## Native XAI и diagnostics

LIME (SLIC, 50 superpixels, 1000 perturbations, seed 2026) и predicted-class Grad-CAM (`layer4.2.conv3`) сохранены как разные native evidence. Registered positive-support L1 discrepancy — отдельный native-XAI diagnostic, а не `system_Gamma`; EYE_F=RET135OD имеет валидный frozen maximum 1.0 при finite/non-zero maps. Disc/cup/rim energy, pointing and top-support overlaps — spatial localization diagnostics, не causal ground truth. Expert1↔Expert2 Dice/CDR отражает вариативность предметной аннотации, а не ошибку эксперта. Faithfulness uses a frozen 10% support mask and 20 random equal-area controls. Its mixed/negative results are retained: spatial localization and perturbation faithfulness are not interchangeable criteria, and no map is therefore called universally faithful.

## Public FuzzyXAI и controls

Full public `FuzzyXAI.wrap(...).explain_one(...)` artifacts exist for 13 selected aliases (physical duplicates are explicitly mapped). `system_Gamma` is canonical alignment from the registered probability-to-technical-risk interface, not LIME↔Grad-CAM agreement. Reduction is not supported by the frozen plan, so chapter-facing Δ is ‘не применялось’. Controls are factual controlled integrity injections; numeric rho is retained, while a critical registered rupture fail-closes final action. The frozen plan declares Grad-CAM required, therefore CONTROL_1 is documented as a missing-required source rather than retrospectively relabelled optional.

## Boundary of system explainability

EYE_D (false positive) and EYE_E (false negative) are kept as model-error cases. A route may be internally consistent and receive a non-blocking technical action while its prediction is wrong: FuzzyXAI controls registered evidence integrity, uncertainty and provenance, not external diagnostic truth without a reference-verification channel. Conversely checkpoint, target and patient-linkage control faults test the fail-closed route policy. The suspect cohort is only descriptive behavior of the frozen binary model on excluded clinically ambiguous cases; no accuracy, FP/FN, sensitivity or specificity is assigned to it.

## Limitations

PAPILA has a small glaucoma class, one public dataset and expert ROI for a controlled experiment; it is not multicenter clinical validation. LIME is a local surrogate and Grad-CAM has coarse spatial resolution. Overlap with disc/cup structures is not causal validation. Technical FuzzyXAI actions are not clinical decisions.

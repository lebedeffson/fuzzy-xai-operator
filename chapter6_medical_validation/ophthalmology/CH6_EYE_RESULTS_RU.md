# CH6-EYE-2 — PAPILA: glaucoma, LIME и public FuzzyXAI

Статус: **реальный открытый PAPILA v2 run, промежуточное закрытие до полного
набора case/controls**. Это воспроизводимое продолжение опубликованной линии
глаукома + LIME, а не репликация чужого checkpoint или клиническая система.

## Данные и протокол

Официальный Figshare article `14798004`, version 2, сообщает лицензию
`GPL 3.0+`. Верифицированы 244 пациента и 488 fundus JPEG; eye labels:
healthy=333, glaucoma=87, suspect=68. Контуров: expert 1=977, expert 2=986;
неполнота отражается в data-root manifest, а не замещается ожидаемым числом.

Primary binary protocol исключает полностью каждого пациента, у которого хотя
бы один глаз имеет `suspect`: 210 clean binary patients используются в
patient-level five-fold CV; 34 suspect-associated patients оставлены только
для отдельного ambiguity cohort. Все глаза пациента закреплены за одним fold.
Манифест CV seed 2026: `d2caccba592639f7a6f2d5e80b7ebb2473a399f6a987ad3ba356f502b6efa926`.

## Модельные результаты

Первичный fixed-seed-2026 запуск ResNet50 с ImageNet initialization и
expert-1 optic-disc ROI завершён для пяти outer folds. Mean ± SD:

| Metric | Value |
|---|---:|
| Accuracy | 0.7690 ± 0.0343 |
| Balanced accuracy | 0.5990 ± 0.0718 |
| F1 (glaucoma) | 0.3256 ± 0.1954 |
| AUROC | 0.6874 ± 0.0637 |
| AUPRC | 0.3954 ± 0.1063 |
| Specificity | 0.8935 ± 0.0676 |
| Sensitivity | 0.3046 ± 0.1735 |

Это ограниченная performance на зарегистрированных outer test folds; она не
подтверждает клиническую безопасность или переносимость. Fold 5 назначен
canonical до измерения test metrics. Для него три seed runs сохранены; seed
2026 выбран только по минимальному internal validation loss.

## Публичные explanation cases

Fold-5 cases `RET028OD` (healthy) и `RET014OD` (glaucoma) получены только
через `FuzzyXAI.wrap(...).explain_one(...)`. LIME (primary) и Grad-CAM
(secondary) хранятся как отдельные native spatial evidence. `system_Gamma`
имеет scope: *canonical FuzzyXAI system alignment; not native-XAI-to-XAI
agreement*. Поэтому spatial LIME/Grad-CAM comparison не называется Gamma.

`RET028OD`: predicted healthy, `u_M=0.0339`, `I_pre=0.8812`,
`rho=0.0339`, action `accept`. `RET014OD`: predicted glaucoma,
`u_M=0.6019`, `I_pre=0.7026`, `rho=0.4250`, action `lower_confidence`.
В обоих случаях Delta имеет status `not_applied`, поэтому chapter-facing
display должен быть `—`, а не измеренное «0.0».

## Verbalizer и ограничения

Pinned local strict SLM `Qwen/Qwen2.5-0.5B-Instruct`
`@7ae557604adf67be50417f59c2c2f167def9a775` запущен только на public
human layer двух случаев. Для обоих `H=0`; SLM не получил raw fundus image,
clinical metadata или право формулировать диагноз.

До финального three-domain claim остаются: полный pre-registered набор
selected cases and controls, spatial segmentation diagnostics, faithfulness
diagnostics и separate suspect auxiliary outputs. Raw images, clinical metadata
и checkpoints не входят в Git или reviewer bundle.

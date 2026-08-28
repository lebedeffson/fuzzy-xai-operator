# Глава 6 — мозг: Allen CCF 2017

## Протоколы

`brain_v1_pilot` сохранён как отдельный исходный пилот (95 patches; 7 held-out patches) и не удаляется. Основной chapter-ready результат — заранее зафиксированный `brain_v2_confirmatory`: config SHA256 `09f564706b13b0457ba6cd82f4544a6d8e5a0d770f04fb5500831c9ec70dbcb0`; 64×64 patches, HPF positives (HPF fraction ≥0.40) и hard gray-matter negatives в той же/близкой coronal section, section-block split с adjacent-block protection. Это protocol improvement, определённый до открытия v2 test results, а не настройка по v1 accuracy.

## Данные и модель

Отдельный v2 manifest SHA256 `2f325d72adebca104cc3abbb7e088fca067364ca8bea5bfb1ceb67608e429990` содержит 1819 patches: train=1231, validation=280, test=308; независимых test section blocks=10. Использован Allen CCF 2017 (25 µm), single-atlas anatomical task OTHER gray matter / HPF; это не clinical disease generalization. InceptionV3 обучена в трёх независимых seed runs. seed=2026: accuracy=0.9773, F1=0.9782, AUROC=0.9991; seed=2027: accuracy=0.9838, F1=0.9845, AUROC=0.9988; seed=2028: accuracy=0.9935, F1=0.9938, AUROC=0.9998. Canonical run `brain-v2-inceptionv3-seed-2027` (seed=2027) выбран только по minimum validation loss=0.019654; его held-out test: accuracy=0.9838, F1=0.9845, AUROC=0.9988, ECE=0.0130, n=308.

## XAI и public FuzzyXAI

Native maps — Grad-CAM (`Mixed_7c`) и full fixed-target logit-space IG. Их `GradCAM_IG_disagreement_diagnostic` и overlap with HPF mask являются пространственными diagnostic quantities, не system Γ и не причинным доказательством. Public `FuzzyXAI.wrap(...).explain_one(...)` создаёт system evidence: registered probability→technical-risk T_ij, `system_Gamma`, uncertainty, I_pre, rho, audit и directed provenance. For this plan reduction is `not_applied`, therefore chapter tables show Δ as «не применялось», not a measured zero loss. Plan has w_p=0: HPF model probability is not clinical risk.

## Cases, controls и ограничения

Selected cases are determined from frozen canonical calibrated test predictions and factual technical metadata before native-XAI execution. BRAIN_G preprocessing mismatch and BRAIN_H checkpoint mismatch are controlled integrity injections: numeric rho stays available but a critical override blocks final action. BRAIN_C/BRAIN_D show that even a wrong anatomical classification may retain an internally consistent route and technical accept without an external truth-verification channel. Results are bounded to one atlas, section-block sampling, this architecture and registered ExplainPlan; they do not establish cross-atlas or clinical transfer.

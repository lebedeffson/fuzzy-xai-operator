# Глава 6 — ЭКГ: PTB-XL

## Статус и постановка

Это независимая проверка FuzzyXAI на 12-канальной ЭКГ, не продолжение работ Аверкина по ЭКГ и не ICU/alarm-постановка. Использован официальный PTB-XL v1.0.3, низкочастотный `records100` (100 Hz, 12 отведений, 1000 отсчётов). Когорта: 20943 записей после детерминированного отбора: NORMAL=0 только при единственном диагностическом superclass NORM; ABNORMAL=1 при MI/STTC/CD/HYP; смешанный NORM+abnormal исключён. Официальные folds 1–8/9/10 использованы как train/validation/test; patient overlap отвергается проверкой.

## Модель и метрики

ECGResNet1D обучалась с weighted cross entropy, AdamW и early stopping по validation loss. Три seed-run: 2026, 2027, 2028. Canonical run — `ecg-resnet1d-seed-2026`, выбран только по minimum validation loss (0.292695); temperature scaling (`T=1.039309`) обучен только на validation fold. На независимом test fold: accuracy=0.8657, balanced accuracy=0.8686, F1=0.8773, AUROC=0.9414, AUPRC=0.9598, NLL=0.3039, Brier=0.0962, ECE(15)=0.0120.

## Объяснение и системный маршрут

Primary native evidence — fixed-target logit-space Integrated Gradients с baseline `zero standardized train mean`; дополнительный experiment-side source — signed temporal occlusion (50 samples/0.5 s). Их common 12×20 diagnostic representation хранится отдельно как `IG_occlusion_disagreement_diagnostic` и **не переименуется в system Γ**. `system_Gamma` — canonical FuzzyXAI system alignment после registered probability→technical-risk `T_ij`, а не agreement IG↔temporal occlusion. `P(ABNORMAL)` — вероятность класса модели, не риск сердечно-сосудистого события. Для этого ExplainPlan rules=`not_applicable`, а reduction=`not_applied`; в chapter-facing таблицах Δ отображается как «не применялось», а не как измеренная нулевая loss.

## Cases, integrity и training evidence

`ECG_A–G` выбраны детерминированно из frozen canonical test predictions: correct NORMAL/ABNORMAL, boundary, highest-confidence FP/FN, maximum diagnostic IG/occlusion disagreement и lowest technical quality. `ECG_H–J` — controlled fault injection: missing waveform provenance, checkpoint mismatch и attribution-target mismatch. Во всех control artifacts numeric rho сохранена, но critical override переводит final action в `block`.

Same-run training artifact для `ptbxl-8` относится к `ecg-resnet1d-seed-2026` и checkpoint `best.pt:epoch:10;sha256:9c1442b9ac7f01feef38ef57e70318bb2524b6b687f9f0a822f7fc7758571f4b`. История включает 10 реально измеренных эпох; first learned=None; forgetting=[]; stability=1.0; loss status=measured. Это конкретная validation probe trajectory, а не утверждение о всей популяции.

## Граница системной объяснимости

`ECG_D` — уверенная ложноположительная, а `ECG_E` — уверенная ложноотрицательная модельная классификация; оба маршрута остаются internally consistent и получают technical `accept`. Это не баг FuzzyXAI: system operator контролирует согласованность, неопределённость, provenance и integrity объяснительного маршрута, но не является oracle истинной диагностической метки. Без внешнего verification/reference channel в ExplainPlan согласованная уверенная ошибка модели может быть неотличима от согласованного корректного решения. Напротив, `ECG_H–J` демонстрируют обнаружение именно controlled integrity faults.

## Ограничения

Фильтрация диагностических superclass не является клинической разметкой событий. Технические actions ExplainPlan не являются клиническими решениями. XAI map — evidence о локальной чувствительности модели; она не доказывает физиологическую причинность. Spatial/temporal diagnostic agreement не является Γ.

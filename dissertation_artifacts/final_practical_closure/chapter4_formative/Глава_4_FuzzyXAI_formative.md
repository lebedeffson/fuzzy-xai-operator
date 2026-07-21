# Глава 4. Практический контур FuzzyXAI

> FORMATIVE ONLY. Финальные статистические выводы появятся только после sealed confirmatory run.

## 4.1 Практическая постановка

Раздел определяет operationally invalid automatic action как практическую цель. Независимая подтверждающая выборка не открыта. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.2 Данные и sealed protocol

Использованы только development/controlled данные; confirmatory identities и labels отсутствуют у tuning runner. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.3 Архитектура практического контроллера

Контроллер разделён на hard structural guard, predictive-risk estimator, route-risk estimator и budgeted optimizer. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.4 Корректная taxonomy аналогов

Post-hoc explainers, glass-box predictors и action policies сравниваются в разных семействах. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.5 H1 и H2

Ранее замороженные H1 и H2 сохраняются без изменения статуса. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.6 H4

Ранее замороженный H4 сохраняется без изменения статуса. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.7 H3-original

H3-original остаётся not_supported; новый практический H3 не переименовывает этот результат. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.8 H3 practical

Formative budget comparison выполнен, но положительный confirmatory claim запрещён. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.9 H5-S и H5-A

H5-S сохранён; H5-A измерен только на controlled faults, natural failures ещё не подтверждены. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.10 H5-P

H5-P-original остаётся not_supported; route validity не объявляется предиктором ошибки модели. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.11 H6-A и H6-B

H6-A измеряет detectability envelope; H6-B имеет статус not_run_requires_two_sealed_independent_datasets. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.12 H7-A и H7-B

Canonical evidence и пользовательская projection разделены; H7-A проверяет hash, H7-B требует independent confirmation. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.13 H8

Компонентная сетка проверена формативно в заранее заданных конфигурациях. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.14 H9

Масштабирование относится к operator layer; стоимость local explainer учитывается отдельно. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.15 Формирующая проверка карточек

AI formative run 2: not_imported; AI-review не является экспертной оценкой. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.16 Воспроизводимость

Каждый formative experiment содержит protocol, manifests, JSONL, Parquet, statistics, claim status и SHA256SUMS. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

## 4.17 Итоговые claims и ограничения

До protocol lock разрешены только технические и formative формулировки; human/expert/domain-safety claims исключены. [evidence:FORMATIVE-SUMMARY sha256:3df7861fcec5de8011f4a50de314660e8a80cf4eef382825235020e7f6c95bfe]

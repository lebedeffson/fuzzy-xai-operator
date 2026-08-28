# Глава 6 — предметный strict SLM verbalizer

Для выполненных ECG и brain-v2 cases использована одна локально закреплённая модель `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` в strict mode. Модель получила только certified claims из public `HumanExplanation`, а не raw ECG/image/model evidence. Выполнено 11 deterministic generations; accepted strict outputs with H=0: 11. Для каждого результата сохранены claim IDs, pinned revision, generation settings и prompt/profile SHA.

Проверяются preservation, а не литературная «красота»: P_fact, H (новые assertions), P_num, P_action и P_lim. Strict output может быть rejected/fallback; такие статусы не скрываются. IDRiD не запускался, поскольку data status=MISSING_DATA.

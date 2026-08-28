# Глава 6 — strict SLM verbalizer

For PAPILA the same pinned local strict backend as ECG and Allen was used: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. It receives only certified public HumanExplanation claims, never raw image or model internals. H=0 for every generated accepted output; rejected/fallback status would be retained instead of hidden. The table preserves P_fact, H, P_num, P_action and P_lim rather than judging literary quality.

## EYE_A / RET038OS

Технический certified text:

Модель определила: здоровый глаз.

Strict ophthalmology text:

Модель определила: здоровый глаз.

## EYE_B / RET098OS

Технический certified text:

Модель определила: глаукома.

Strict ophthalmology text:

- Модель определила: глаукома.
- Не все проверки подтверждены: Для части проверки не хватает подтверждённых данных, поэтому автоматическое применение ограничено.
- Причины прогноза не раскрыты: Доступен итог модели, но нет подтверждённых данных о конкретных признаках, правилах или примерах, которые его поддержали.
- Доверие ограничивают: не все проверки подтверждены, причины прогноза не раскрыты. Не хватает данных для проверок: model internals, model rules or concepts. Эти ограничения не позволяют использовать результат автоматически.
- Передать результат предметному специалисту и проверить исходные данные, основные причины и ограничения модели.

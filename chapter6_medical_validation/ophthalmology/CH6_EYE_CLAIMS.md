# Claim–evidence map (pre-run)

| Claim | Source/artifact | Metric | Allowed wording | Forbidden overclaim |
|---|---|---|---|---|
| Published line uses fundus classification and explainability | `SOURCES.md` | bibliographic record | «эксперимент продолжает опубликованную постановку» | «точная репликация» |
| Public runtime preserves an eye-case route | future `cases/*/result.json` | typed route status | «FuzzyXAI сформировал проверяемый системный маршрут» | «клинически подтвердил диагноз» |
| Attribution overlaps lesion markup | future `tables/E5_*` | lesion energy/pointing/IoU | «пространственное соответствие разметке» | «причинная верность» |
| Missing provenance changes route status | future control artifact | U_trace/status/action | «обнаружена неполнота трассы» | «модель ошиблась медицински» |
| Target/checkpoint mismatch is blocked | future control artifact | critical override | «обнаружено системное несоответствие» | «обнаружена патология» |
| Strict verbalizer preserves claims | future `tables/E8_*` | P_fact/H/P_num/P_action/P_lim | «не добавил неподдержанных утверждений на выбранных cases» | «прошёл экспертную клиническую оценку» |

Все численные claims остаются `MISSING_DATA` до реального run.

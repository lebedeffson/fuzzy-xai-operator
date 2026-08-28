# Медицинская практическая валидация FuzzyXAI

Research-only контур главы 6 проверяет один frozen P19 contract на трёх
заранее определённых модальностях: IDRiD fundus RGB, PTB-XL 12-канальный
временной сигнал и Allen Mouse Brain Atlas Nissl. Фактически завершены две
public-runtime validation: PTB-XL и отдельный `brain_v2_confirmatory`.
IDRiD остаётся `MISSING_DATA`: официальный interactive access не заменяется
зеркалом или выдуманным результатом. Это не заявление о трёх эмпирически
выполненных доменах. Каталог не входит в installable wheel.

Статусы данных и запусков всегда evidence-first: недоступный набор остаётся
`MISSING_DATA`, а не заменяется зеркалом или синтетическим результатом.

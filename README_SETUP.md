# FuzzyXAI v5 — установка и проверка научного контура

Проект-основание: `lebedeffson/fuzzy-xai-operator`, main `aae7aa717bff913240cbb0026807cd1ccfa67160` (27.08.2026).
Это **overlay** поверх полного исходного репозитория, а не готовая замена репозитория.
В текущем окружении нет доступа к записи GitHub и не выполнен интеграционный запуск на полном checkout.

## Применение

Создайте чистую рабочую ветку в локальном клоне. Распакуйте архив в корень репозитория, сохраняя пути файлов. Для безопасности резервная копия ветки/чистый `git status` обязательны. Исходные модули не заменяются распаковкой: их исправляет установщик.

```bash
git switch -c feature/fuzzyxai-doctoral-v5
python install_fuzzyxai_upgrade.py --check
python install_fuzzyxai_upgrade.py
python install_fuzzyxai_upgrade.py --check
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_doctoral_*.py tests/test_release_transactional_installer.py
python -m pytest -q tests/test_chapter5_*.py
python -m experiments.doctoral_native_repeated --out reports/chapter5/native_fuzzy_repeated
python -m pytest -q
```

Установщик атомарно применяет **восемь** локальных правок; проверка проводится до записи, при ошибке изменения откатываются. Повторный запуск идемпотентен. Старые сценарии `apply_doctoral_upgrade.py` и `apply_ch5_practice.py` оставлены для обратной совместимости, их не следует выполнять отдельно поверх единого установщика.

## Оценка результата

- `docs/DOCTORAL_CONFORMANCE_1_TO_4.md` — матрица реализации математических контрактов.
- `docs/DOCTORAL_V5_CHAPTER5_RESULTS_RU.md` — методика, результаты, ограничения WDBC.
- `.github/workflows/doctoral-conformance.yml` и `doctoral-native-evidence.yml` — проверки для полной GitHub CI (не исполнялись в данном окружении).
- `RELEASE_MANIFEST.json` — SHA256 и точный перечень файлов пакета.

Контрольный набор ЭКГ — технический fixture, а не клиническая выборка. PTB-XL для полной проверки качества отведений не запускался. Никакие метрики WDBC не относятся к ЭКГ.

Локально проверены 114 изолированных тестов; регрессия исходного фреймворка, внедрение в приложениях и эксплуатационная валидация сайта остаются отдельными обязательными шагами. Не называйте сертификат проверкой правильности медицинского диагноза.

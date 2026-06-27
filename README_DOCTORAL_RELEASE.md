# FuzzyXAI Doctoral Release

## Требования

- Python 3.10+
- LibreOffice или `soffice` для DOCX render gate
- Git

## Установка

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Если editable extras недоступны:

```bash
pip install -r requirements.txt
```

## Финальная проверка

```bash
make doctorate-release-check
```

Команда воспроизводит HYBRID-XIRIS batch, proof package, таблицы главы 5, DOCX gates, audit package, runtime release package и audit tests.

## HYBRID-XIRIS batch

```bash
python -m fuzzyxai.run_scenario hybrid_xiris --batch
```

Выходы:

- `reports/studio_batch/hybrid_xiris_batch_cases.csv`
- `reports/studio_batch/hybrid_xiris_batch_summary.json`
- `reports/studio_batch/hybrid_xiris_proof_package.json`

## Таблицы главы 5

```bash
python -m fuzzyxai.export_tables --scenario hybrid_xiris
```

Выходы:

- `reports/chapter5/studio_tables/table_5_2_explainplan.csv`
- `reports/chapter5/studio_tables/table_5_3_membership.csv`
- `reports/chapter5/studio_tables/table_5_4_dE.csv`
- `reports/chapter5/studio_tables/table_5_5_run_summary.csv`
- `reports/chapter5/studio_tables/table_5_6_risk_decomposition.csv`

## Пользовательский слой

```bash
make studio
```

Откройте адрес, который напечатает NiceGUI. Эталонный сценарий: HYBRID-XIRIS.

## Smoke UI

```bash
make studio-smoke
```

Проверяется наличие ключевых смысловых элементов: `БЛОКИРОВКА`, `γ = 0.351`, `Δ = 0.106811`, `ρ = 0.800`, `D_quality_source_conflict`, скрытый технический след.

## Главы 4-5

Проверяемые DOCX лежат здесь:

- `docs/chapters/glava_4_FuzzyXAI_corrected_final.docx`
- `docs/chapters/glava_5_FuzzyXAI_corrected_final.docx`

DOCX gates:

```bash
python -m fuzzyxai.audit.docx_chapters
python -m fuzzyxai.audit.docx_format
python -m fuzzyxai.audit.docx_render_gate
```

## Пакеты

- `fuzzyxai_final_audit_package.zip` — audit/artifact package
- `fuzzyxai_doctoral_runtime_release.zip` — runtime source release
- `visual_artifacts_latest.zip` — visual artifacts and reports

## Финальный PASS

Ожидаемый статус:

```text
BLOCKER: 0
MAJOR: 0
MINOR: 0
Formula reference gate: PASS
DOCX content gate: PASS
DOCX XML style gate: PASS
DOCX visual render gate: PASS
```

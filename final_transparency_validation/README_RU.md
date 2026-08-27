# FuzzyXAI P19 — пакет независимой проверки

P19 закрывает системный оператор FuzzyXAI в каноническом публичном runtime.
Системные операторные scientific fields создаются одним
`FuzzyXAI.wrap(...).explain_one(...)`; system generators только сохраняют
public reports, audit, serialization, inspect и visualization. Image generator
дополнительно формирует caller-defined connected-component masks из публичного
тензора атрибуций; это региональное evidence, а не системная математика P19.

## Основные артефакты

- `P19_RUNTIME_SUMMARY_RU.md` — краткие итоговые числа.
- `P19_CHANGELOG.md` — закрытые контракты и удалённые shortcuts.
- `semantic_audit.md` — ручная трассировка значений, формул и inputs.
- `scientific_alignment_report.md` — соответствие операторной математике.
- `golden_system_accept/` — complete system route, action=accept.
- `golden_system_conflict/` — controlled trace fault injection, numeric rho
  preserved, critical action=block.
- `golden_system_reduction/` — real held-out RF disagreement, non-degenerate
  interval and positive Delta.
- `golden_tabular/` — real StandardScaler/LogisticRegression local chain;
  incomplete full rho disclosed as `None`, partial score named separately.
- `golden_training/` — real same-run SGD partial-fit history and final model.
- `golden_image_28x28/` — full Fashion-MNIST IG tensor, overlay and measured
  logit-space completeness.
- `full_regression_log.txt` — 1261 passed, 11 skipped, 643 warnings in
  288.12 s.
- `wheel_smoke_test_log.txt` and `wheel/` — source-independent installed
package acceptance.

Wheel содержит `fuzzyxai.experiments`, потому что семь защищённых операторов
manifest используют callables этого namespace. Шесть отсоединённых research
namespaces исключены; NiceGUI доступен только через optional extra `ui`.

## Scientific boundaries

- RandomForest global importance is never presented as a local reason.
- Gamma requires an executed `T_ij`.
- Delta belongs to real uncertainty-representation reduction.
- `U_model`, `U_rules`, and `U_trace` independently aggregate into `u_M`.
- `I_pre` belongs to typed `E_pre`.
- rho is the strict five-component function; missing non-zero-weight inputs
  yield `rho=None`, not hidden renormalization.
- Critical rupture overrides action without erasing numeric rho.
- Missing, optional and not-applicable evidence remain distinct.

## Reproduction

Run the six public-runtime generators with the framework on `PYTHONPATH`:

```bash
python final_transparency_validation/generate_p19_system_cases.py accept
python final_transparency_validation/generate_p19_system_cases.py conflict
python final_transparency_validation/generate_p19_system_cases.py reduction
python final_transparency_validation/generate_golden_tabular.py
python final_transparency_validation/generate_golden_training.py
python final_transparency_validation/generate_golden_image_28x28.py
```

The wheel-only smoke is in `wheel_smoke.py`; execute it from a directory
outside the repository after installing only the wheel.

## Known limitations

- Image IG convergence is measured, with final 512-step absolute residual
  0.0015958548 (relative 0.0001606005); it is not claimed to be exactly zero.
- Accept/conflict reductions are lossless; the separate reduction case records
  positive Delta from a real non-degenerate interval.
- The conflict case is deliberate trace fault injection.
- Optional model ecosystems absent from the validation environment are skipped
  explicitly in the regression log.

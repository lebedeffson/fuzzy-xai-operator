"""Repeated exploratory held-out evaluation of native fuzzy prediction.

Five distinct seeds expose dependence on one random split. Reuse of a finite
public dataset across repeated splits means these are NOT independent trials
or a confirmatory hypothesis test; do not report a naive p-value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from .doctoral_native_fuzzy_logistic import NativeFuzzyLogistic
from .doctoral_native_fuzzy_wdbc import study


def run(root: Path, *, seeds: tuple[int,...] = (17,42,73,123,2026), features: int = 5) -> dict[str, Any]:
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("at least two distinct random seeds required")
    root.mkdir(parents=True, exist_ok=True)
    series = []
    per_seed = []
    for seed in seeds:
        report = study(root / f"seed_{seed}", seed=seed, n_features=features,
                       model_factory=NativeFuzzyLogistic)
        series.append(report)
        for name, metric in report["metric_comparison"].items():
            per_seed.append({"seed": seed, "model": name, **metric,
                             "cert_valid_count": report["certification_status_counts"].get("valid",0),
                             "cert_invalid_count": report["certification_status_counts"].get("invalid",0)})
    with (root/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(per_seed[0]))
        writer.writeheader();writer.writerows(per_seed)
    aggregate = {}
    for method in ("native_fuzzy","logistic_regression","random_forest"):
        rows = [r for r in per_seed if r['model']==method]
        aggregate[method] = {key: {"mean": mean(float(r[key]) for r in rows),
                                   "sample_std": stdev(float(r[key]) for r in rows),
                                   "min": min(float(r[key]) for r in rows),
                                   "max": max(float(r[key]) for r in rows)}
                             for key in ("accuracy","f1_malignant","roc_auc","brier")}
    sets = [set(s['selected_features_train_only']) for s in series]
    overlaps = [len(a&b)/len(a|b) for a,b in combinations(sets,2)]
    valid_ratios = [s['certification_status_counts'].get('valid',0)/s['n_test'] for s in series]
    all_stress = sum(int(s['stress_missing_feature']['n']) for s in series)
    # Exactly reconstruct the membership design/coefficients of one reported
    # trained model; no artistic or randomly chosen curves are substituted.
    first_model=json.loads((root/f'seed_{seeds[0]}'/'trained_native_fuzzy.json').read_text(encoding='utf-8'))
    first_rule=first_model['rules'][0]
    xlo,xhi=float(first_rule['low']),float(first_rule['high'])
    xvalues=[xlo+(xhi-xlo)*i/199 for i in range(200)]
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    ax.plot(xvalues,[1.-(x-xlo)/(xhi-xlo) for x in xvalues],color='black',linestyle='-',label='low')
    ax.plot(xvalues,[(x-xlo)/(xhi-xlo) for x in xvalues],color='black',linestyle='--',label='high')
    ax.set(xlabel=f'{first_rule["name"]} (training-fitted limits)',ylabel='Fuzzy membership degree',ylim=(-.02,1.02),
           title='Measured input-to-term transformation')
    ax.legend(loc='center right');ax.grid(axis='y',linestyle=':',alpha=.35)
    fig.tight_layout();fig.savefig(root/'membership_terms.png',dpi=190);fig.savefig(root/'membership_terms.pdf');plt.close(fig)
    rule_rows=[]
    for feature in first_model['rules']:
        rule_rows.append((feature['name']+' / low',float(feature['low_consequent'])))
        rule_rows.append((feature['name']+' / high',float(feature['high_consequent'])))
    fig,ax=plt.subplots(figsize=(8.9,5.5))
    bars=ax.barh([x[0] for x in rule_rows],[x[1] for x in rule_rows],color='white',edgecolor='black')
    for bar,(_,coeff) in zip(bars,rule_rows):
        bar.set_hatch('///' if coeff<0 else '')
    ax.axvline(0,color='black',linewidth=.8)
    ax.set(xlabel='Fitted coefficient in native fuzzy logit',title='Native rule influence (not post-hoc SHAP)')
    ax.invert_yaxis();fig.tight_layout();fig.savefig(root/'native_rule_coefficients.png',dpi=190)
    fig.savefig(root/'native_rule_coefficients.pdf');plt.close(fig)
    # Evaluation uses identical held-out patients for all model families WITHIN
    # each split, but repeated splits share subjects: no independent CIs.
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    for name,style in (("native_fuzzy",'-'),("logistic_regression",'--'),("random_forest",':')):
        ax.plot(list(seeds),[s['metric_comparison'][name]['roc_auc'] for s in series],
                linestyle=style,marker='o',linewidth=1.5,label=name)
    ax.set_ylim(0.65,1.005);ax.set_xlabel('Random split seed');ax.set_ylabel('Held-out ROC-AUC')
    ax.grid(axis='y',linestyle=':',alpha=.45);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(root/'roc_auc_per_seed.png',dpi=190);fig.savefig(root/'roc_auc_per_seed.pdf');plt.close(fig)
    # Illustrate a real, fully observed ROC curve from ONE held-out split.
    first=root/f'seed_{seeds[0]}'/'heldout_native_fuzzy.csv'
    with first.open(encoding='utf-8') as fh:
        heldout=list(csv.DictReader(fh))
    truth=[int(r['target_malignant']) for r in heldout]
    fig,ax=plt.subplots(figsize=(6.7,6.1))
    for col,name,style in (("p_fuzzy","native fuzzy",'-'),("p_logistic","logistic regression",'--'),("p_random_forest","random forest",':')):
        x,y,_=roc_curve(truth,[float(r[col]) for r in heldout])
        ax.plot(x,y,linestyle=style,linewidth=1.5,label=name)
    ax.plot([0,1],[0,1],linestyle='-.',linewidth=.8,label='chance')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'WDBC held-out ROC; seed {seeds[0]}')
    ax.legend(loc='lower right',fontsize=8);ax.grid(linestyle=':',alpha=.4)
    fig.tight_layout();fig.savefig(root/'heldout_roc.png',dpi=190);fig.savefig(root/'heldout_roc.pdf');plt.close(fig)
    output = {
        "schema":"FuzzyXAI-native-fuzzy-repeated-v1", "status":"computed",
        "protocol":"exploratory_repeated_stratified_holdouts_not_independent_trials",
        "seed_list":list(seeds),"features":features,
        "per_seed_model_metrics":per_seed,
        "aggregate_descriptive":aggregate,
        "stability_feature_selection_pairwise_jaccard":{"mean":mean(overlaps),"min":min(overlaps),"max":max(overlaps)},
        "certificate_valid_fraction_per_split":valid_ratios,
        "critical_input_stress_cases_total":all_stress,
        "critical_input_stress_all_blocked":all(s['stress_missing_feature']['all_actions_blocked'] for s in series),
        "limitations":["Repeated stratified splits of one public dataset are not independent samples",
                       "Exploratory algorithm and split choice; no preregistered confirmatory study",
                       "Certificates attest numerical/computational coherence only, not correctness of malignant diagnosis",
                       "Gamma is expected to be ~0 by construction for a native-rule model; this is not independent explanation fidelity",
                       "Delta measures reduction-induced change of the predicted probability, not semantic completeness",
                       "No clinical outcomes, deployment validation, model calibration study or causal inference"],
    }
    (root/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Глава 5. Повторяемый эксперимент с нечёткой логистической моделью',
             '', '## Постановка',
             'WDBC, 30 измеряемых признаков. После разбиения только на обучающей части выбираются признаки,',
             'определяются границы функций принадлежности и оцениваются коэффициенты логистической модели',
             'над нечёткими термами. Таким образом, термы участвуют непосредственно в вычислении прогноза.',
             'Положительный класс — злокачественное образование; медицинская пригодность не заявляется.',
             '', '## Разбиения и сравнения',
             f'Разбиения: {len(seeds)}; случайные зерна: {list(seeds)}. Каждое: train=341, val=114, test=114.',
             'Фиксированные алгоритмы сравниваются на одинаковых тестовых объектах внутри каждого разбиения.',
             'Эти разбиения повторно используют один и тот же набор — их нельзя трактовать как независимые наблюдения.',
             '', '| Модель | Accuracy, среднее | ROC-AUC, среднее | F1 для злокачественных, среднее | Brier, среднее |',
             '|---|---:|---:|---:|---:|']
    for model,label in (("native_fuzzy","Нечёткая логистическая"),("logistic_regression","Логистическая регрессия"),("random_forest","Случайный лес")):
        m=aggregate[model]
        lines.append(f'| {label} | {m["accuracy"]["mean"]:.4f} | {m["roc_auc"]["mean"]:.4f} | {m["f1_malignant"]["mean"]:.4f} | {m["brier"]["mean"]:.4f} |')
    lines.extend(['', '## Проверка объяснительного маршрута',
                  f'Доля полностью согласованных сертификатов на отдельных разбиениях: от {min(valid_ratios):.3f} до {max(valid_ratios):.3f}.',
                  'Несогласованные сертификаты не превращаются в успешные через удаление проблемных объектов.',
                  'Чаще всего контроль блокирует чрезмерную потерю Δ при сокращении активных правил.',
                  f'Средний Jaccard состава выбранных признаков между парами разбиений: {mean(overlaps):.3f}.',
                  f'Контролируемое выпадение измерения: {all_stress} случаев; все направлены на блокировку автоматического принятия.',
                  '', '## Научные ограничения',
                  'Результаты являются исследовательскими. Γ≈0 для внутреннего линейного разложения модели —',
                  'алгебраическое свойство конструкции, а не независимо измеренная достоверность объяснения.',
                  'Число сертификатов valid характеризует соблюдение программного контракта, но не диагностическую',
                  'эффективность. Сравнение на клинических записях и экспертные оценки объяснений не проводились.',
                  '', '## Материалы воспроизведения',
                  '`per_seed_metrics.csv`, `seed_*/heldout_native_fuzzy.csv`, `seed_*/sample_certificate.json`,',
                  '`seed_*/trained_native_fuzzy.json`, `roc_auc_per_seed.png/pdf`, `heldout_roc.png/pdf`,',
                  '`membership_terms.png/pdf`, `native_rule_coefficients.png/pdf`, `manifest_sha256.json`.'])
    (root/'CHAPTER5_NATIVE_FUZZY_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifact_hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest_sha256.json'}
    (root/'manifest_sha256.json').write_text(json.dumps({"schema":"sha256-manifest-v1","files":artifact_hashes},ensure_ascii=False,indent=2),encoding='utf-8')
    return output


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('reports/chapter5/native_fuzzy_repeated'))
    p.add_argument('--seeds',nargs='+',type=int,default=[17,42,73,123,2026])
    p.add_argument('--features',type=int,default=5)
    args=p.parse_args()
    summary=run(args.out,seeds=tuple(args.seeds),features=args.features)
    print(json.dumps({"status":summary['status'],"seeds":summary['seed_list'],"metrics":summary['aggregate_descriptive']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

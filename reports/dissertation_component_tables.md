# Полные таблицы по компонентам системы FuzzyXAI

## 1) Компоненты системы

| Этап | Объект | Вход | Выход | Метрика | Код | Пояснение |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset | DatasetCase | raw/local dataset mode | normalized case data | status, rows, domain | fuzzyxai/datasets/*, apps/services/layered_case.py | Подготовка данных в единый формат для последующих слоёв. |
| Model | prediction, p(y=1), uncertainty | features(x) | prediction, predicted_risk | accuracy, roc_auc | fuzzyxai/pipelines/dataset_observer_pipeline.py | Базовый прогноз без принятия финального решения. |
| Omega | E_model=<L,mu,R,alpha,u,tau> | model output | structured explanation object | u, active rules | fuzzyxai/core/system_operator.py | Перевод численного прогноза в проверяемую объяснительную структуру. |
| Expl / Rupture | T_ij or D_ij | E_model -> E_risk -> E_action | morphism / rupture | gamma, gamma_max, chi_R | fuzzyxai/core/composition.py, fuzzyxai/core/diagnostics.py | Проверка согласованности переходов между слоями. |
| Fuzzy Representation | F0 / F_int / F_H / F_N_src / FML-audit | uncertainty profile | selected_class, Delta | coverage, complexity, reduction_loss | fuzzyxai/risk/representation_selection.py, apps/services/layered_case.py | Выбор минимально достаточного класса неопределенности. |
| Context Topos | RiskContext, AutoAccept, chi_Auto | context presheaves | auto apply allowed/denied | AutoAccept(E) | fuzzyxai/category/context_topos.py, fuzzyxai/category/subpresheaf.py | Контекстно-зависимый фильтр для безопасного автоприменения. |
| Risk Observer | rho, chi_R, chi_R^crit, action | risk components | accept/lower/request/defer/block | mean_rho, action_distribution | fuzzyxai/risk/*, apps/layered_demo.py | Интегральный риск и пороговая политика безопасного действия. |
| Reports | JSON/CSV/MD | pipeline artifacts | dissertation tables | reproducibility | experiments/dataset_benchmark.py, experiments/dissertation_demo_summary.py | Фиксация воспроизводимых результатов для защиты. |

## 2) Датасеты и готовность

| Dataset mode | Domain | Status | Rows | What validates |
| --- | --- | --- | --- | --- |
| breast_cancer | medical | READY | 569 | Medical risk baseline: probability + observer action policy. |
| diabetes_binary | medical | READY | 442 | Borderline medical uncertainty: lower_confidence/request_more_data behavior. |
| wine_risk | general | READY | 178 | Cross-domain robustness check beyond one medical dataset. |
| synthetic_ruptures | controlled | READY | 900 | Controlled rupture generation for Expl/HoTT and safe-action diagnostics. |
| registry_programs | tabular_text | READY | 10007 | Text+tabular explainability/ranking scenario from CIT registry. |
| registry_mosmed_doctor_analysis | medical_audit | READY | 10 | Medical audit scenario: doctor/model decision alignment. |
| registry_steel_ir | industrial_cv | READY | 8080 | Industrial control scenario: transfer beyond medical domain. |

## 3) Метрики по датасетам (benchmark)

| Dataset | Acc | ROC AUC | Observer action acc | mean_I_pre | mean_rho | rupture_rate |
| --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | 0.972027972027972 | 0.9949685534591195 | 0.5594405594405595 | 0.7935691880178595 | 0.2076536833161616 | 0.0 |
| diabetes_binary | 0.7567567567567568 | 0.8298701298701298 | 0.0990990990990991 | 0.765687631853013 | 0.38605707865721867 | 0.0 |
| wine_risk | 0.9777777777777777 | 1.0 | 0.6222222222222222 | 0.7902422015533586 | 0.192412638958542 | 0.0 |
| synthetic_ruptures | 0.9511111111111111 | 0.9955908289241623 | 0.3022222222222222 | 0.7728543248175938 | 0.3172559460867956 | 0.0 |
| registry_programs | 0.5011990407673861 | 0.4948539968984046 | 0.0 | 0.7605362825571323 | 0.46119131706650957 | 0.0 |
| registry_mosmed_doctor_analysis | 1.0 | nan | 0.6666666666666666 | 0.7607764309603002 | 0.19313492184381179 | 0.0 |
| registry_steel_ir | 0.9920792079207921 | 0.5 | 1.0 | 0.7893476847952753 | 0.06749594132621858 | 0.0 |

## 4) Кейсы наблюдателя по слоям

| Scenario | predicted_risk | uncertainty | selected_class | Delta | I_pre | rho | chi_R | chi_R_crit | action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| safe | 0.000531 | 0.001061 | FML-audit | 0.07 | 0.901666 | 0.005341 | 0 | 0 | accept |
| ambiguous | 0.514468 | 0.971064 | FML-audit | 0.07 | 0.610124 | 0.8 | 1 | 1 | block |
| rupture | 0.514468 | 0.971064 | FML-audit | 0.07 | 0.610124 | 0.8 | 1 | 1 | block |

## 5) Пороговая политика

| Условие | Действие | Пояснение |
| --- | --- | --- |
| chi_R^crit(x)=1 | block | критический разрыв |
| rho(x)>=0.80 and chi_R^crit(x)=0 | defer_to_human | высокий риск без крит. разрыва |
| rho(x)<0.80 and chi_R(x)=1 and chi_R^crit(x)=0 | request_more_data | некритический разрыв |

Примечание: для MosMed оригинальный архив ~104.9GB, в локальном контуре используется малый табличный аудит-слепок.
"""Generates the golden_tabular/ case: a real StandardScaler->LogisticRegression
pipeline on Breast Cancer Wisconsin, with real training history, a real
reference corpus, real similar cases (including a genuine counterexample,
with real query/reference values and deltas), a real counterfactual, and
full Russian domain language for all 30 features. Nothing here is
hand-drawn — every artifact is produced by the library from this one
explain_one() call.

P17: no artificial second explanatory channel is added anymore — Γ requires
a genuine second channel (e.g. a fuzzy/rule model's native activations),
which this bare linear pipeline does not have. This case honestly stays at
its real explanation level rather than fabricating E5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework" / "fuzzyxai"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fuzzyxai import FuzzyXAI, ObservationContext
from fuzzyxai.core.explain_plan import ExplainPlan
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "golden_tabular"
OUT.mkdir(exist_ok=True)

# P17 item 8: complete Russian labels for all 30 real features — no
# feature is left as its raw English sklearn name.
FEATURE_LABELS_RU = {
    "mean radius": "средний радиус",
    "mean texture": "средняя текстура",
    "mean perimeter": "средний периметр",
    "mean area": "средняя площадь",
    "mean smoothness": "средняя гладкость контура",
    "mean compactness": "средняя компактность",
    "mean concavity": "средняя вогнутость контура",
    "mean concave points": "среднее число вогнутых точек контура",
    "mean symmetry": "средняя симметрия",
    "mean fractal dimension": "средняя фрактальная размерность",
    "radius error": "стандартная ошибка радиуса",
    "texture error": "стандартная ошибка текстуры",
    "perimeter error": "стандартная ошибка периметра",
    "area error": "стандартная ошибка площади",
    "smoothness error": "стандартная ошибка гладкости контура",
    "compactness error": "стандартная ошибка компактности",
    "concavity error": "стандартная ошибка вогнутости контура",
    "concave points error": "стандартная ошибка числа вогнутых точек",
    "symmetry error": "стандартная ошибка симметрии",
    "fractal dimension error": "стандартная ошибка фрактальной размерности",
    "worst radius": "худший (максимальный) радиус",
    "worst texture": "худшая (максимальная) текстура",
    "worst perimeter": "худший (максимальный) периметр",
    "worst area": "худшая (максимальная) площадь",
    "worst smoothness": "худшая (максимальная) гладкость контура",
    "worst compactness": "худшая (максимальная) компактность",
    "worst concavity": "худшая (максимальная) вогнутость контура",
    "worst concave points": "худшее (максимальное) число вогнутых точек контура",
    "worst symmetry": "худшая (максимальная) симметрия",
    "worst fractal dimension": "худшая (максимальная) фрактальная размерность",
}


def build_domain_language(feature_names: list[str]) -> dict:
    missing = [name for name in feature_names if name not in FEATURE_LABELS_RU]
    if missing:
        raise ValueError(f"missing Russian labels for features: {missing}")
    return {
        "features": {name: {"label": FEATURE_LABELS_RU[name]} for name in feature_names},
        "classes": {
            "0": {"label": "злокачественная опухоль (malignant)"},
            "1": {"label": "доброкачественная опухоль (benign)"},
        },
        "actions": {},
    }


def main() -> None:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])

    object_index = 12  # a real object for which find_tabular_counterfactuals genuinely finds a class-changing neighbor
    x_object = X_test.iloc[object_index].to_numpy()
    pipe.fit(X_train, y_train)

    train_ids = [f"train_{i}" for i in range(len(X_train))]
    domain_language = build_domain_language(list(X.columns))
    plan = ExplainPlan(domain_language=domain_language)

    context = ObservationContext(
        reference_data=X_train.to_numpy(), reference_labels=y_train.tolist(), reference_ids=train_ids,
        dataset_version="breast_cancer_wisconsin_sklearn_v1",
    )
    fx = FuzzyXAI.wrap(pipe, explain_plan=plan, observation_context=context)

    result = fx.explain_one(
        x_object,
        object_id="p0",
        feature_names=list(X.columns),
        include_similar_cases=True,
        include_counterfactuals=True,
        include_training_trace=False,
    )

    (OUT / "full_report_reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (OUT / "full_report_audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    (OUT / "summary.txt").write_text(result.summary("user", detail="full"), encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(result.to_dict(detail="audit"), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "compact.json").write_text(json.dumps(result.to_dict(detail="compact"), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(result.view_model.explanation_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    explanation_level = result.explanation_level
    missing = result.missing_channels
    limitations_lines = [
        f"explanation_level = {explanation_level}",
        f"missing_channels = {list(missing)}",
        f"gamma = {result.view_model.disagreement.get('gamma')}",
        f"delta = {result.view_model.disagreement.get('delta')}",
        f"reduction_status = {result.view_model.disagreement.get('reduction_status')}",
        f"rho = {result.view_model.risk.get('rho')}",
        f"risk_status = {result.view_model.risk.get('status')}",
        f"action = {result.action}",
        "",
        "P17: этот случай НЕ содержит искусственного второго объяснительного",
        "канала. Линейная модель имеет ровно один локальный канал объяснения",
        "(численные вклады), поэтому Γ честно не измеряется автоматически —",
        "сравнивать не с чем. Уровень объяснения ниже — это корректный",
        "результат, а не недоработка.",
        "Δ также честно не вычисляется автоматически: reconstruction_error",
        "(ошибка восстановления линейной формулы) — это отдельная quality-метрика,",
        "а не потеря при реальной редукции представления, которой здесь не было.",
        "ρ помечен как incomplete, если ExplainPlan ожидает компоненту",
        "(например, uncertainty), для которой нет реального источника —",
        "он НЕ перенормируется молча в уверенный accept.",
    ]
    (OUT / "limitations.txt").write_text("\n".join(limitations_lines) + "\n", encoding="utf-8")

    try:
        result.visualize(view="object_representation", output=str(OUT / "object_representation.png"))
        # P17: focused provenance (5-10 nodes), not a sample of the full
        # 80+-node graph. One default (action-focused) and one anchored on
        # the linear-reconstruction claim specifically, matching the
        # worked example (dataset -> preprocessor -> transformed value ->
        # coefficient -> contribution -> claim -> prediction).
        result.visualize(view="provenance", output=str(OUT / "provenance_action.png"))
        linear_claim = next((c for c in result.claims if c.claim_type == "linear_reconstruction"), None)
        if linear_claim is not None:
            result.visualize(view="provenance", selector=f"claim:{linear_claim.claim_id}", output=str(OUT / "provenance_claim.png"))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - exporter records renderer failures
        (OUT / "visualization_error.txt").write_text(str(exc), encoding="utf-8")

    print("explanation_level:", explanation_level)
    print("missing_channels:", missing)
    print("action:", result.action)
    print("gamma:", result.view_model.disagreement.get("gamma"))
    print("delta:", result.view_model.disagreement.get("delta"))
    print("rho:", result.view_model.risk.get("rho"))
    print("risk_status:", result.view_model.risk.get("status"))


if __name__ == "__main__":
    main()

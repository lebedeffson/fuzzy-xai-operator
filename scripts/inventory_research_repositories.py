from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OWNERS = ["fims9000", "lebedeffson"]
REGISTRY_DIR = ROOT / "registry"
REPORT_DIR = ROOT / "reports" / "validation" / "repository_inventory"
SITE_DATA = ROOT / "site" / "dubnaxai" / "src" / "data"
SELECTED_STATUSES = {"required", "research_candidate"}


EXCLUDE_OVERRIDES: dict[str, str] = {
    "fims9000/fims9000": "profile repository, not a research/application source",
    "lebedeffson/Avalonia_Mnist": "education/demo repository, not part of DubnaXAI research layer",
    "lebedeffson/fuzzy-xai-operator": "current monorepo, tracked as local source rather than external research repo",
    "lebedeffson/lebedeffson": "profile repository, not a research/application source",
    "lebedeffson/MobilePhotoSensor": "mobile utility/project repo; excluded until research relevance is proven",
    "lebedeffson/SYNT_ISIC": "duplicate/empty-side listing; canonical selected repo is fims9000/SYNT_ISIC",
    "lebedeffson/TMPKWebApp": "web/hackathon utility repo, not part of DubnaXAI research layer",
    "lebedeffson/Yandex_skill": "assistant skill/utility repo, not part of DubnaXAI research layer",
}


ROLE_OVERRIDES: dict[str, dict[str, str]] = {
    "fims9000/XAI-2.0-SHAP-regularized-ANFIS": {
        "research_area_ru": "SHAP-регуляризованный ANFIS и встроенная интерпретируемость",
        "site_role": "карточка метода и модели",
        "framework_role": "табличный нейро-нечёткий адаптер",
        "application_role": "сценарий gd_anfis_shap",
        "scenario_id": "gd_anfis_shap",
        "adapter": "AnfisShapAdapter",
        "status": "required",
    },
    "fims9000/Trust-ADE": {
        "research_area_ru": "оценка доверия к ИИ и динамическая объяснимость",
        "site_role": "карточка метода доверия",
        "framework_role": "внешний сравнительный модуль метрик доверия",
        "application_role": "сопоставление Trust-ADE и FuzzyXAI",
        "scenario_id": "trust_ade_comparison",
        "adapter": "TrustAdeAdapter",
        "status": "required",
    },
    "fims9000/XAI-CausalLayered": {
        "research_area_ru": "многоуровневая нейро-символьная архитектура XAI 2.0",
        "site_role": "карточка архитектурного метода",
        "framework_role": "пример многоуровневого маршрута F_ML",
        "application_role": "сценарий causal_layered",
        "scenario_id": "causal_layered",
        "adapter": "LayeredXaiAdapter",
        "status": "required",
    },
    "fims9000/CAS_ISIC": {
        "research_area_ru": "исследовательская классификация и сегментация дерматологических изображений",
        "site_role": "карточка медицинской исследовательской модели",
        "framework_role": "адаптер медицинского изображения",
        "application_role": "сценарий cas_isic_medical_image",
        "scenario_id": "cas_isic_medical_image",
        "adapter": "MedicalImageAdapter",
        "status": "required",
    },
    "fims9000/SYNT_ISIC": {
        "research_area_ru": "генерация ISIC, Time-SHAP и каузальные интервенции",
        "site_role": "карточка генеративного XAI-инструмента",
        "framework_role": "адаптер временной атрибуции и интервенций",
        "application_role": "сценарий synt_isic_time_shap",
        "scenario_id": "synt_isic_time_shap",
        "adapter": "TimeShapInterventionAdapter",
        "status": "required",
    },
    "fims9000/BeaconXAI": {
        "research_area_ru": "контрсвидетельства и аудит временных объяснений",
        "site_role": "карточка метода BEACON-XAI",
        "framework_role": "адаптер контрсвидетельств",
        "application_role": "сценарий beacon_xai",
        "scenario_id": "beacon_xai",
        "adapter": "BeaconXaiAdapter",
        "status": "required",
    },
    "fims9000/Groundwater-Risk": {
        "research_area_ru": "геориск и оценка подземных вод",
        "site_role": "карточка гео-рискового сценария",
        "framework_role": "адаптер геориска",
        "application_role": "сценарий groundwater_risk",
        "scenario_id": "groundwater_risk",
        "adapter": "GeoRiskAdapter",
        "status": "research_candidate",
    },
    "fims9000/KAN-XAI-2.0-System": {
        "research_area_ru": "KAN и XAI 2.0",
        "site_role": "карточка метода KAN-XAI",
        "framework_role": "пример адаптера KAN-объяснений",
        "application_role": "сценарий kan_xai",
        "scenario_id": "kan_xai",
        "adapter": "KanXaiAdapter",
        "status": "research_candidate",
    },
    "fims9000/anza_lira": {
        "research_area_ru": "прикладной исследовательский прототип",
        "site_role": "карточка проекта после review",
        "framework_role": "только после анализа входов/выходов",
        "application_role": "candidate scenario",
        "scenario_id": "anza_lira",
        "adapter": "TBD",
        "status": "research_candidate",
    },
    "lebedeffson/hybrid-xiris-biometric": {
        "research_area_ru": "биометрическая идентификация по радужной оболочке",
        "site_role": "карточка модели и основной прикладной сценарий",
        "framework_role": "пример адаптера изображения и операторного маршрута",
        "application_role": "сценарий hybrid_xiris",
        "scenario_id": "hybrid_xiris",
        "adapter": "IrisAdapter",
        "status": "required",
    },
    "lebedeffson/deep-neuro-fuzzy": {
        "research_area_ru": "глубокие нейро-нечёткие модели и Routed KAFN",
        "site_role": "карточка фреймворка глубоких нейро-нечётких моделей",
        "framework_role": "адаптер DeepNeuroFuzzy/KAFN",
        "application_role": "сценарий deep_neuro_fuzzy_kafn",
        "scenario_id": "deep_neuro_fuzzy_kafn",
        "adapter": "DeepNeuroFuzzyAdapter",
        "status": "required",
    },
    "lebedeffson/FuzzyAttentionNetworks": {
        "research_area_ru": "мультимодальная классификация с fuzzy attention",
        "site_role": "карточка метода нечёткого внимания",
        "framework_role": "адаптер attention weights и membership functions",
        "application_role": "сценарий fuzzy_attention_multimodal",
        "scenario_id": "fuzzy_attention_multimodal",
        "adapter": "FuzzyAttentionAdapter",
        "status": "required",
    },
    "lebedeffson/RuFLEX": {
        "research_area_ru": "гибридная платформа глубокого нечёткого обучения для геоданных",
        "site_role": "карточка платформы RuFLEX",
        "framework_role": "гео-нечёткий адаптер",
        "application_role": "сценарий ruflex_geo",
        "scenario_id": "ruflex_geo",
        "adapter": "RuFlexAdapter",
        "status": "required",
    },
    "lebedeffson/bonner-shap-reconstruction": {
        "research_area_ru": "SHAP-реконструкция и объяснительные вклады",
        "site_role": "карточка метода реконструкции",
        "framework_role": "адаптер SHAP-реконструкции",
        "application_role": "сценарий shap_reconstruction",
        "scenario_id": "shap_reconstruction",
        "adapter": "ShapReconstructionAdapter",
        "status": "required",
    },
    "lebedeffson/eaar-regularization": {
        "research_area_ru": "регуляризация атрибуций EAAR",
        "site_role": "карточка метода регуляризации",
        "framework_role": "внешний модуль атрибутивной регуляризации",
        "application_role": "сценарий eaar_regularization",
        "scenario_id": "eaar_regularization",
        "adapter": "EaarAdapter",
        "status": "required",
    },
    "lebedeffson/competency-assessment-system-dubna": {
        "research_area_ru": "оценка компетенций и рекомендательные системы",
        "site_role": "карточка образовательной системы",
        "framework_role": "табличный/рекомендательный адаптер после review",
        "application_role": "сценарий competency_assessment",
        "scenario_id": "competency_assessment",
        "adapter": "CompetencyAssessmentAdapter",
        "status": "research_candidate",
    },
}


def fetch_repos(owner: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/users/{owner}/repos?per_page=100&sort=full_name"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def repo_id(owner: str, repo: str) -> str:
    return f"{owner}_{repo}".lower().replace("-", "_").replace(".", "_")


def yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def write_yaml(items: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Auto-generated by scripts/inventory_research_repositories.py",
        "# Source: GitHub public repositories for fims9000 and lebedeffson.",
        "",
    ]
    for item in items:
        lines.append(f"- id: {yaml_scalar(item['id'])}")
        for key, value in item.items():
            if key == "id":
                continue
            lines.append(f"  {key}: {yaml_scalar(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize_repo(owner: str, repo: dict[str, Any]) -> dict[str, Any]:
    full = f"{owner}/{repo['name']}"
    role = ROLE_OVERRIDES.get(full, {})
    license_info = repo.get("license") or {}
    status = role.get("status", "profile_or_utility_review")
    selected = status in SELECTED_STATUSES
    return {
        "id": repo_id(owner, repo["name"]),
        "owner": owner,
        "repo": repo["name"],
        "url": repo["html_url"],
        "description": repo.get("description") or "",
        "language": repo.get("language") or "",
        "license": license_info.get("spdx_id") or "",
        "research_area_ru": role.get("research_area_ru", "требует научной классификации"),
        "site_role": role.get("site_role", "карточка репозитория после review"),
        "framework_role": role.get("framework_role", "нет прямой зависимости от FuzzyXAI до review"),
        "application_role": role.get("application_role", "нет сценария до review"),
        "integration_mode": "external_repo_adapter" if status in {"required", "research_candidate"} else "catalog_only",
        "adapter": role.get("adapter", "TBD"),
        "scenario_id": role.get("scenario_id", ""),
        "status": status,
        "selected_for_dubnaxai": selected,
        "selection_reason": (
            role.get("research_area_ru", "selected research/application repository")
            if selected
            else EXCLUDE_OVERRIDES.get(full, "public repository kept only in inventory until manual research review")
        ),
        "license_review_required": True,
        "commit_pin_required": True,
        "clone_path": f"external/research_repos/{owner}/{repo['name']}",
        "default_branch": repo.get("default_branch") or "",
        "pushed_at": repo.get("pushed_at") or "",
    }


def write_reports(all_items: list[dict[str, Any]], selected_items: list[dict[str, Any]], excluded_items: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owners": OWNERS,
        "discovered_count": len(all_items),
        "selected_count": len(selected_items),
        "excluded_count": len(excluded_items),
        "selected_repositories": selected_items,
        "excluded_repositories": excluded_items,
        "all_repositories": all_items,
    }
    (REPORT_DIR / "repository_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "repository_excluded.json").write_text(json.dumps(excluded_items, ensure_ascii=False, indent=2), encoding="utf-8")
    (SITE_DATA / "repositories.json").write_text(json.dumps(selected_items, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Research Repository Inventory",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Discovered repositories: `{len(all_items)}`",
        f"Selected research/application repositories: `{len(selected_items)}`",
        f"Excluded/catalog-only repositories: `{len(excluded_items)}`",
        "",
        "Only public repositories with a research/application role are exported to the DubnaXAI site data and `registry/repositories.yaml`.",
        "Profile, mobile, assistant, web-utility and duplicate repositories remain in the audit inventory only.",
        "",
        "## Selected research/application repositories",
        "",
        "| Owner | Repo | Status | Scenario | Adapter | License |",
        "|---|---|---|---|---|---|",
    ]
    for item in selected_items:
        lines.append(
            f"| {item['owner']} | [{item['repo']}]({item['url']}) | {item['status']} | "
            f"{item['scenario_id']} | {item['adapter']} | {item['license']} |"
        )
    lines.extend([
        "",
        "## Excluded from DubnaXAI research layer",
        "",
        "| Owner | Repo | Status | Reason |",
        "|---|---|---|---|",
    ])
    for item in excluded_items:
        lines.append(f"| {item['owner']} | [{item['repo']}]({item['url']}) | {item['status']} | {item['selection_reason']} |")
    (REPORT_DIR / "repository_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    items: list[dict[str, Any]] = []
    for owner in OWNERS:
        for repo in fetch_repos(owner):
            items.append(normalize_repo(owner, repo))
    items.sort(key=lambda item: (item["owner"], item["repo"].lower()))
    selected_items = [item for item in items if item["selected_for_dubnaxai"]]
    excluded_items = [item for item in items if not item["selected_for_dubnaxai"]]
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    write_yaml(selected_items, REGISTRY_DIR / "repositories.yaml")
    write_reports(items, selected_items, excluded_items)
    print(REGISTRY_DIR / "repositories.yaml")
    print(REPORT_DIR / "repository_inventory.md")


if __name__ == "__main__":
    main()

"""Build Chapter 6 tables, figures and chapter-ready reports from public artifacts.

This exporter deliberately reads stored public-result/audit artifacts only.  It
does not recompute FuzzyXAI system quantities or create clinical claims.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS, TABLES, FIGURES = (ROOT / name for name in ("reports", "tables", "figures"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = TABLES / name
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    markdown = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    markdown.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def system_row(domain: str, case: str, case_row: dict, audit_path: Path) -> dict[str, object]:
    system = load(audit_path)["system_evidence"]
    uncertainty = system["uncertainty"]
    risk = system["risk"]
    return {
        "domain": domain, "case": case, "object_id": case_row.get("object_id"),
        "truth": case_row.get("label"), "prediction": case_row.get("prediction"),
        "confidence": case_row.get("confidence"), "correct": case_row.get("correct"),
        "system_Gamma": system["gamma"]["gamma"],
        "Gamma_scope": "canonical FuzzyXAI system alignment; not native-XAI-to-XAI agreement",
        "U_model": uncertainty["u_model"],
        "U_rules_status": uncertainty["sources"]["U_rules"]["status"], "U_trace": uncertainty["u_trace"],
        "u_M": uncertainty["u_m"], "Delta_status": system["reduction"]["status"],
        "Delta_display": "—" if system["reduction"]["status"] == "not_applied" else system["reduction"].get("delta"),
        "I_pre": system["i_pre"]["value"], "rho": risk["rho"],
        "candidate_action": risk["candidate_action"], "critical": risk["critical_override"],
        "final_action": risk["action"],
    }


def plot_confusion(path: Path, matrix: list[list[int]], labels: tuple[str, str], title: str) -> None:
    fig, axis = plt.subplots(figsize=(4.6, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row, values in enumerate(matrix):
        for col, value in enumerate(values): axis.text(col, row, str(value), ha="center", va="center")
    axis.set(xticks=(0, 1), yticks=(0, 1), xticklabels=labels, yticklabels=labels, xlabel="Прогноз", ylabel="Истинный класс", title=title)
    fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def ecg() -> tuple[dict, list[dict], list[dict]]:
    root = ROOT / "ecg_ptbxl"; prepared = Path(__import__("os").environ["FUZZYXAI_CH6_DATA_ROOT"]) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    runs = [(load(path), path.parent) for path in sorted((root / "outputs" / "runs").glob("*/run.json"))]
    canonical, _canonical_dir = min(runs, key=lambda item: (min(row["validation_loss"] for row in item[0]["history"]), -max(row["validation_auroc"] for row in item[0]["history"]), item[0]["seed"]))
    selected = load(root / "outputs" / "cases" / "selected_cases.json")["cases"]
    cases: list[dict] = []
    for case, row in selected.items():
        if "object_id" in row:
            cases.append(system_row("ECG", case, row, root / "outputs" / "cases" / str(row["object_id"]) / "audit.json"))
    controls = load(root / "outputs" / "controls" / "controls_summary.json")
    control_rows = [{"domain": "ECG", "fault": key, "expected": "critical override -> block", "observed": value["final_action"], "pass": value["critical_override"] and value["final_action"] == "block"} for key, value in controls.items()]
    metrics_rows = []
    for run, _ in runs:
        metric = run["test_metrics_calibrated"]
        metrics_rows.append({"domain": "ECG", "seed": run["seed"], "run_id": run["run_id"], "accuracy": metric["accuracy"], "f1": metric["f1"], "auroc": metric["auroc"], "auprc": metric["auprc"], "ece": metric["ece_15_bin"], "nll": metric["nll"]})
    test = canonical["test_metrics_calibrated"]
    plot_confusion(FIGURES / "ECG_04_confusion_matrix.png", test["confusion_matrix"], ("NORMAL", "ABNORMAL"), "PTB-XL: canonical test fold")
    labels = np.load(prepared / "labels.npy"); folds = np.load(prepared / "folds.npy")
    fig, axis = plt.subplots(figsize=(7, 4)); names = ("train", "validation", "test"); values = [labels[folds <= 8], labels[folds == 9], labels[folds == 10]]
    axis.bar(names, [int((value == 0).sum()) for value in values], label="NORMAL"); axis.bar(names, [int((value == 1).sum()) for value in values], bottom=[int((value == 0).sum()) for value in values], label="ABNORMAL")
    axis.set(title="PTB-XL primary cohort: official folds", ylabel="Число записей"); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "ECG_03_folds.png", dpi=170); plt.close(fig)
    representative = selected["ECG_B"]["object_id"]; case_dir = root / "outputs" / "cases" / representative
    ig, occ = np.load(case_dir / "integrated_gradients_signed.npy"), np.load(case_dir / "temporal_occlusion_signed.npy")
    for name, data, title in (("ECG_06_ig.png", ig, "Integrated Gradients: 12 отведений × 1000 отсчётов"), ("ECG_07_occlusion.png", occ, "Temporal occlusion: 12 отведений × 20 окон")):
        fig, axis = plt.subplots(figsize=(10, 4)); image = axis.imshow(data, aspect="auto", cmap="coolwarm"); axis.set(title=title, ylabel="Отведение", xlabel="Время / окно"); fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(FIGURES / name, dpi=170); plt.close(fig)
    from chapter6_medical_validation.ecg_ptbxl.src.xai import common_ig_representation, common_occlusion_representation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7));
    for axis, data, title in zip(axes, (common_ig_representation(ig), common_occlusion_representation(occ)), ("T_IG → 12×20", "T_OCC → 12×20"), strict=True):
        image = axis.imshow(data, aspect="auto", cmap="coolwarm"); axis.set_title(title); fig.colorbar(image, ax=axis)
    fig.tight_layout(); fig.savefig(FIGURES / "ECG_08_common_representation.png", dpi=170); plt.close(fig)
    copy2(case_dir / "provenance_action.png", FIGURES / "ECG_09_provenance.png")
    fig, axis = plt.subplots(figsize=(8, 4)); shown = [case for case in cases if case["case"] in {"ECG_A", "ECG_B", "ECG_C", "ECG_D", "ECG_E", "ECG_G"}]; x = np.arange(len(shown)); axis.bar(x - .25, [float(row["system_Gamma"]) for row in shown], .25, label="system Γ"); axis.bar(x, [float(row["u_M"]) for row in shown], .25, label="u_M"); axis.bar(x + .25, [float(row["rho"]) for row in shown], .25, label="ρ"); axis.set(xticks=x, xticklabels=[str(row["case"]) for row in shown], title="PTB-XL selected public cases"); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "ECG_10_system_metrics.png", dpi=170); plt.close(fig)
    copy2(root / "outputs" / "controls" / "ECG_I_checkpoint_mismatch" / "provenance_action.png", FIGURES / "ECG_11_checkpoint_fault.png")
    training = load(root / "outputs" / "training" / "ptbxl-8" / "training_summary.json")
    report = f"""# Глава 6 — ЭКГ: PTB-XL

## Статус и постановка

Это независимая проверка FuzzyXAI на 12-канальной ЭКГ, не продолжение работ Аверкина по ЭКГ и не ICU/alarm-постановка. Использован официальный PTB-XL v1.0.3, низкочастотный `records100` (100 Hz, 12 отведений, 1000 отсчётов). Когорта: {len(labels)} записей после детерминированного отбора: NORMAL=0 только при единственном диагностическом superclass NORM; ABNORMAL=1 при MI/STTC/CD/HYP; смешанный NORM+abnormal исключён. Официальные folds 1–8/9/10 использованы как train/validation/test; patient overlap отвергается проверкой.

## Модель и метрики

ECGResNet1D обучалась с weighted cross entropy, AdamW и early stopping по validation loss. Три seed-run: {', '.join(str(row['seed']) for row in metrics_rows)}. Canonical run — `{canonical['run_id']}`, выбран только по minimum validation loss ({min(row['validation_loss'] for row in canonical['history']):.6f}); temperature scaling (`T={canonical['calibration']['temperature']:.6f}`) обучен только на validation fold. На независимом test fold: accuracy={test['accuracy']:.4f}, balanced accuracy={test['balanced_accuracy']:.4f}, F1={test['f1']:.4f}, AUROC={test['auroc']:.4f}, AUPRC={test['auprc']:.4f}, NLL={test['nll']:.4f}, Brier={test['brier']:.4f}, ECE(15)={test['ece_15_bin']:.4f}.

## Объяснение и системный маршрут

Primary native evidence — fixed-target logit-space Integrated Gradients с baseline `zero standardized train mean`; дополнительный experiment-side source — signed temporal occlusion (50 samples/0.5 s). Их common 12×20 diagnostic representation хранится отдельно как `IG_occlusion_disagreement_diagnostic` и **не переименуется в system Γ**. `system_Gamma` — canonical FuzzyXAI system alignment после registered probability→technical-risk `T_ij`, а не agreement IG↔temporal occlusion. `P(ABNORMAL)` — вероятность класса модели, не риск сердечно-сосудистого события. Для этого ExplainPlan rules=`not_applicable`, а reduction=`not_applied`; в chapter-facing таблицах Δ отображается как «не применялось», а не как измеренная нулевая loss.

## Cases, integrity и training evidence

`ECG_A–G` выбраны детерминированно из frozen canonical test predictions: correct NORMAL/ABNORMAL, boundary, highest-confidence FP/FN, maximum diagnostic IG/occlusion disagreement и lowest technical quality. `ECG_H–J` — controlled fault injection: missing waveform provenance, checkpoint mismatch и attribution-target mismatch. Во всех control artifacts numeric rho сохранена, но critical override переводит final action в `block`.

Same-run training artifact для `ptbxl-8` относится к `{training['training_run_id']}` и checkpoint `{training['final_checkpoint_ref']}`. История включает {len(training['history'])} реально измеренных эпох; first learned={training['first_learned_epoch']}; forgetting={training['forgetting_events']}; stability={training['stability']}; loss status={training['loss_status']}. Это конкретная validation probe trajectory, а не утверждение о всей популяции.

## Граница системной объяснимости

`ECG_D` — уверенная ложноположительная, а `ECG_E` — уверенная ложноотрицательная модельная классификация; оба маршрута остаются internally consistent и получают technical `accept`. Это не баг FuzzyXAI: system operator контролирует согласованность, неопределённость, provenance и integrity объяснительного маршрута, но не является oracle истинной диагностической метки. Без внешнего verification/reference channel в ExplainPlan согласованная уверенная ошибка модели может быть неотличима от согласованного корректного решения. Напротив, `ECG_H–J` демонстрируют обнаружение именно controlled integrity faults.

## Ограничения

Фильтрация диагностических superclass не является клинической разметкой событий. Технические actions ExplainPlan не являются клиническими решениями. XAI map — evidence о локальной чувствительности модели; она не доказывает физиологическую причинность. Spatial/temporal diagnostic agreement не является Γ.
"""
    (REPORTS / "CH6_ECG_RESULTS_RU.md").write_text(report, encoding="utf-8")
    return canonical, cases, control_rows


def brain() -> tuple[dict, list[dict], list[dict]]:
    root = ROOT / "brain_allen"; prepared = Path(__import__("os").environ["FUZZYXAI_CH6_DATA_ROOT"]) / "brain" / "allen_ccf_25um" / "prepared"
    runs = [(load(path), path.parent) for path in sorted((root / "outputs" / "runs").glob("*/run.json"))]
    canonical, _canonical_dir = min(runs, key=lambda item: (min(row["validation_loss"] for row in item[0]["history"]), item[0]["seed"]))
    selected = load(root / "outputs" / "cases" / "selected_cases.json")["cases"]
    cases = [system_row("BRAIN", case, row, root / "outputs" / "cases" / str(row["object_id"]) / "audit.json") for case, row in selected.items() if "object_id" in row]
    controls = load(root / "outputs" / "controls" / "controls_summary.json")
    control_rows = [{"domain": "BRAIN", "fault": key, "expected": "critical override -> block", "observed": value["final_action"], "pass": value["critical_override"] and value["final_action"] == "block"} for key, value in controls.items()]
    test = canonical["test_metrics_calibrated"]
    plot_confusion(FIGURES / "BRAIN_05_confusion_matrix.png", test["confusion_matrix"], ("OTHER", "HPF"), "Allen CCF test blocks")
    item = selected["BRAIN_A"]; case_dir = root / "outputs" / "cases" / item["object_id"]
    patch = np.load(prepared / "patches.npy", mmap_mode="r")[int(item["prepared_index"])]
    cam, ig = np.load(case_dir / "grad_cam_raw.npy"), np.load(case_dir / "integrated_gradients_signed.npy").sum(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5));
    for axis, image, title, cmap in zip(axes, (patch, cam, ig), ("Nissl patch", "Grad-CAM", "Integrated Gradients"), ("gray", "magma", "coolwarm"), strict=True): axis.imshow(image, cmap=cmap); axis.set(title=title); axis.axis("off")
    fig.tight_layout(); fig.savefig(FIGURES / "BRAIN_06_patch_gradcam_ig.png", dpi=170); plt.close(fig)
    copy2(case_dir / "provenance_action.png", FIGURES / "BRAIN_07_provenance.png")
    fig, axis = plt.subplots(figsize=(7, 4)); x=np.arange(len(cases)); axis.bar(x-.25,[float(c['system_Gamma']) for c in cases],.25,label="system Γ"); axis.bar(x,[float(c['u_M']) for c in cases],.25,label="u_M"); axis.bar(x+.25,[float(c['rho']) for c in cases],.25,label="ρ"); axis.set(xticks=x,xticklabels=[c['case'] for c in cases],title="Allen selected public cases"); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "BRAIN_08_system_metrics.png", dpi=170); plt.close(fig)
    copy2(root / "outputs" / "controls" / "BRAIN_H_checkpoint_mismatch" / "provenance_action.png", FIGURES / "BRAIN_09_integrity_control.png")
    patches_info = load(prepared / "patches.json")
    report = f"""# Глава 6 — мозг: Allen CCF 2017

## Постановка

Работа продолжает нейроморфологическую линию Колесниковой/Аверкина на открытом Allen Mouse Brain Common Coordinate Framework 2017, а не переиспользует данные исходной публикации. Использованы официальные 25 µm Nissl volume, annotation volume и ontology. Класс HPF построен только по официальной иерархии descendants hippocampal formation; comparator — прочее серое вещество, без background negatives.

## Leakage protection, модель и метрики

Подготовлено {len(patches_info)} patch metadata records; split выполнен section-block-wise, поэтому patch одного блока не пересекает train/validation/test. Canonical InceptionV3 run `{canonical['run_id']}` выбран по minimum validation loss, seed={canonical['seed']}; test blocks n={test['n']}. Наблюдаемые test metrics: accuracy={test['accuracy']:.4f}, F1={test['f1']:.4f}, AUROC={test['auroc']:.4f}, ECE={test['ece_15_bin']:.4f}. Малый тестовый объём и один atlas делают эти показатели демонстрационными, не обобщающими clinical claims.

## Native XAI и FuzzyXAI

Native maps — Grad-CAM (`Mixed_7c`) и full fixed-target logit-space IG. Их spatial overlap с HPF annotation и `GradCAM_IG_disagreement_diagnostic` — diagnostic only: не system Γ и не причинное доказательство. Public `explain_one` строит E_model из binary probability, применяет registered T_ij и сохраняет `system_Gamma`, U-model/trace, I_pre, rho и action. Для spatial representation frozen P19 contract не поддерживает canonical Pi/iota, поэтому reduction is `not_applied`; в chapter-facing представлении Δ обозначается «не применялось», а не измеренной нулевой потерей. У brain ExplainPlan w_p=0: probability HPF не имеет clinical-risk meaning.

## Controls и ограничения

`BRAIN_G` (preprocessing mismatch) и `BRAIN_H` (checkpoint mismatch) — controlled fault-injection scenarios: numeric rho сохраняется, но critical override → block. Они не являются естественно обнаруженными ошибками модели. Сведения об atlas, split, hashes, provenance PNG и reports находятся в `brain_allen/outputs`.
"""
    (REPORTS / "CH6_BRAIN_RESULTS_RU.md").write_text(report, encoding="utf-8")
    return canonical, cases, control_rows


def brain_v2() -> tuple[dict, list[dict], list[dict]]:
    """Project the separately frozen v2 confirmatory outputs into Chapter 6."""

    root = ROOT / "brain_allen" / "outputs_v2_confirmatory"
    prepared = Path(__import__("os").environ["FUZZYXAI_CH6_DATA_ROOT"]) / "brain" / "allen_ccf_25um" / "prepared_v2_confirmatory"
    runs = [(load(path), path.parent) for path in sorted((root / "runs").glob("*/run.json"))]
    canonical, _ = min(runs, key=lambda item: (min(row["validation_loss"] for row in item[0]["history"]), item[0]["seed"]))
    selected = load(root / "cases" / "selected_cases.json")["cases"]
    cases = [system_row("BRAIN_V2", case, row, root / "cases" / str(row["object_id"]) / "audit.json") for case, row in selected.items() if "object_id" in row]
    controls = load(root / "controls" / "controls_summary.json")
    control_rows = [{"domain": "BRAIN_V2", "fault": key, "expected": "critical override -> block", "observed": value["final_action"], "pass": value["critical_override"] and value["final_action"] == "block"} for key, value in controls.items()]
    manifest = load(prepared / "dataset_manifest.json")
    test = canonical["test_metrics_calibrated"]
    plot_confusion(FIGURES / "BRAIN_V2_05_confusion_matrix.png", test["confusion_matrix"], ("OTHER", "HPF"), "Allen CCF v2 confirmatory test blocks")
    item = selected["BRAIN_A"]; case_dir = root / "cases" / item["object_id"]
    patch = np.load(prepared / "patches.npy", mmap_mode="r")[int(item["prepared_index"])]
    cam, ig = np.load(case_dir / "grad_cam_raw.npy"), np.load(case_dir / "integrated_gradients_signed.npy").sum(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, image, title, cmap in zip(axes, (patch, cam, ig), ("Nissl patch", "Grad-CAM", "Integrated Gradients"), ("gray", "magma", "coolwarm"), strict=True):
        axis.imshow(image, cmap=cmap); axis.set(title=title); axis.axis("off")
    fig.tight_layout(); fig.savefig(FIGURES / "BRAIN_V2_06_patch_gradcam_ig.png", dpi=170); plt.close(fig)
    copy2(case_dir / "provenance_action.png", FIGURES / "BRAIN_V2_07_provenance.png")
    fig, axis = plt.subplots(figsize=(7, 4)); x = np.arange(len(cases))
    axis.bar(x - .25, [float(row["system_Gamma"]) for row in cases], .25, label="system Γ")
    axis.bar(x, [float(row["u_M"]) for row in cases], .25, label="u_M")
    axis.bar(x + .25, [float(row["rho"]) for row in cases], .25, label="ρ")
    axis.set(xticks=x, xticklabels=[str(row["case"]) for row in cases], title="Allen v2 selected public cases"); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "BRAIN_V2_08_system_metrics.png", dpi=170); plt.close(fig)
    copy2(root / "controls" / "BRAIN_H_checkpoint_mismatch" / "provenance_action.png", FIGURES / "BRAIN_V2_09_integrity_control.png")
    metrics = "; ".join(f"seed={run['seed']}: accuracy={run['test_metrics_calibrated']['accuracy']:.4f}, F1={run['test_metrics_calibrated']['f1']:.4f}, AUROC={run['test_metrics_calibrated']['auroc']:.4f}" for run, _ in runs)
    report = f"""# Глава 6 — мозг: Allen CCF 2017

## Протоколы

`brain_v1_pilot` сохранён как отдельный исходный пилот (95 patches; 7 held-out patches) и не удаляется. Основной chapter-ready результат — заранее зафиксированный `brain_v2_confirmatory`: config SHA256 `09f564706b13b0457ba6cd82f4544a6d8e5a0d770f04fb5500831c9ec70dbcb0`; 64×64 patches, HPF positives (HPF fraction ≥0.40) и hard gray-matter negatives в той же/близкой coronal section, section-block split с adjacent-block protection. Это protocol improvement, определённый до открытия v2 test results, а не настройка по v1 accuracy.

## Данные и модель

Отдельный v2 manifest SHA256 `{manifest['manifest_sha256']}` содержит {sum(manifest['patch_counts'].values())} patches: train={manifest['patch_counts']['train']}, validation={manifest['patch_counts']['validation']}, test={manifest['patch_counts']['test']}; независимых test section blocks={manifest['independent_section_blocks']['test']}. Использован Allen CCF 2017 (25 µm), single-atlas anatomical task OTHER gray matter / HPF; это не clinical disease generalization. InceptionV3 обучена в трёх независимых seed runs. {metrics}. Canonical run `{canonical['run_id']}` (seed={canonical['seed']}) выбран только по minimum validation loss={min(row['validation_loss'] for row in canonical['history']):.6f}; его held-out test: accuracy={test['accuracy']:.4f}, F1={test['f1']:.4f}, AUROC={test['auroc']:.4f}, ECE={test['ece_15_bin']:.4f}, n={test['n']}.

## XAI и public FuzzyXAI

Native maps — Grad-CAM (`Mixed_7c`) и full fixed-target logit-space IG. Их `GradCAM_IG_disagreement_diagnostic` и overlap with HPF mask являются пространственными diagnostic quantities, не system Γ и не причинным доказательством. Public `FuzzyXAI.wrap(...).explain_one(...)` создаёт system evidence: registered probability→technical-risk T_ij, `system_Gamma`, uncertainty, I_pre, rho, audit и directed provenance. For this plan reduction is `not_applied`, therefore chapter tables show Δ as «не применялось», not a measured zero loss. Plan has w_p=0: HPF model probability is not clinical risk.

## Cases, controls и ограничения

Selected cases are determined from frozen canonical calibrated test predictions and factual technical metadata before native-XAI execution. BRAIN_G preprocessing mismatch and BRAIN_H checkpoint mismatch are controlled integrity injections: numeric rho stays available but a critical override blocks final action. BRAIN_C/BRAIN_D show that even a wrong anatomical classification may retain an internally consistent route and technical accept without an external truth-verification channel. Results are bounded to one atlas, section-block sampling, this architecture and registered ExplainPlan; they do not establish cross-atlas or clinical transfer.
"""
    (REPORTS / "CH6_BRAIN_RESULTS_RU.md").write_text(report, encoding="utf-8")
    return canonical, cases, control_rows


def slm_rows() -> list[dict[str, object]]:
    roots = [("ECG", ROOT / "ecg_ptbxl" / "outputs" / "cases"), ("brain_v2_confirmatory", ROOT / "brain_allen" / "outputs_v2_confirmatory" / "cases")]
    rows: list[dict[str, object]] = []
    for domain, root in roots:
        for path in sorted(root.glob("*/strict_slm.json")):
            value = load(path)
            rows.append({"domain": domain, "case": path.parent.name, **value["metrics"], "status": value["status"], "model": value["model_id"], "revision": value["revision"], "prompt_profile_sha256": value.get("prompt_profile_sha256", "legacy_pending_regeneration")})
    return rows


def top_reports(ecg_cases: list[dict], brain_cases: list[dict], controls: list[dict]) -> None:
    eyes = """# Глава 6 — офтальмология: IDRiD\n\nIDRiD остаётся primary continuation dataset for fundus/diabetic-retinopathy validation. Официальный IEEE DataPort доступ требует аутентифицированного принятия условий и данных локально нет. Поэтому в этом пакете нет выдуманных метрик, prediction, XAI, Gamma, Delta, rho или action для глаза. Реализованный experiment scaffold и precise access blocker находятся в `ophthalmology/DATA_ACCESS_REQUIRED_IDRID.md`.\n"""
    (REPORTS / "CH6_EYES_RESULTS_RU.md").write_text(eyes, encoding="utf-8")
    cross = """# Глава 6 — cross-domain result\n\nОдин public contract `FuzzyXAI.wrap(...).explain_one(...)` использован на PTB-XL 12-lead ECG и Allen CCF 2017 Nissl patches (`brain_v2_confirmatory`). Во всех фактически выполненных routes prediction, native XAI, typed evidence, system evidence, audit и directed provenance получены из public result. Общими остаются semantic invariants; domain-specific остаются preprocessing, adapter, native XAI, ExplainPlan language и factual observation context. IDRiD не включён в assertion of completed three-domain empirical validation, потому что official data access is missing. Raw rho values не сравниваются между ExplainPlans как единая physical scale.\n"""
    (REPORTS / "CH6_CROSS_DOMAIN_RESULTS_RU.md").write_text(cross, encoding="utf-8")
    medical = """# Глава 6 — практические результаты medical validation\n\n## Дизайн\n\nВыполнены две реальные open-data public-runtime validations: PTB-XL (12-lead ECG) и Allen CCF 2017 (`brain_v2_confirmatory`, section-block-wise Nissl patches). `brain_v1_pilot` сохранён как исторический малый pilot и не заменяет confirmatory route. Офтальмологическая линия IDRiD подготовлена как primary continuation, но не выполнена: официальный IEEE DataPort требует аутентифицированного принятия условий, поэтому метрики, XAI и system results для глаз не фабрикуются. Следовательно, текущий результат — две выполненные validation и одна подготовленная application line, а не three-domain empirical claim.\n\n## Что измерялось\n\nДля каждого выполненного domain public `FuzzyXAI.wrap(...).explain_one(...)` возвращал prediction, model-native evidence, typed evidence, `system_Gamma`, uncertainty profile, action, audit и directed provenance. `system_Gamma` обозначает canonical alignment зарегистрированных system interfaces; он не является agreement между двумя native XAI maps. ECG IG/temporal occlusion и brain Grad-CAM/IG сравнения остаются отдельными diagnostic quantities. Когда ExplainPlan указывает `reduction=not_applied`, chapter tables показывают Δ как «не применялось», а не как измеренную нулевую loss.\n\n## Representative outcomes и controls\n\nECG и brain reports содержат real registered test splits, model metrics, selected cases, native XAI and public system artifacts. Controlled checkpoint/provenance/preprocessing faults были обнаружены как critical integrity conditions и fail-closed блокировались. Это не означает, что framework выявляет все ошибки модели: в ECG уверенные ложноположительный и ложноотрицательный predictions могут сохранять internally consistent route и получать technical `accept`, если ExplainPlan не содержит внешнего verification evidence.\n\n## Ограничения\n\nТехническое action policy не является clinical decision support; native attribution не доказывает биологическую причинность; Allen v2 ограничен single-atlas section-block generalization. Все сильные claims ограничены зарегистрированными данными, ExplainPlan и provenance конкретного route.\n"""
    (REPORTS / "CH6_MEDICAL_RESULTS_RU.md").write_text(medical, encoding="utf-8")
    rows = slm_rows()
    generated = [row for row in rows if row["status"] == "generated" and row["H"] == 0]
    verbalizer = f"""# Глава 6 — предметный strict SLM verbalizer\n\nДля выполненных ECG и brain-v2 cases использована одна локально закреплённая модель `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` в strict mode. Модель получила только certified claims из public `HumanExplanation`, а не raw ECG/image/model evidence. Выполнено {len(rows)} deterministic generations; accepted strict outputs with H=0: {len(generated)}. Для каждого результата сохранены claim IDs, pinned revision, generation settings и prompt/profile SHA.\n\nПроверяются preservation, а не литературная «красота»: P_fact, H (новые assertions), P_num, P_action и P_lim. Strict output может быть rejected/fallback; такие статусы не скрываются. IDRiD не запускался, поскольку data status=MISSING_DATA.\n"""
    (REPORTS / "CH6_VERBALIZER_RESULTS_RU.md").write_text(verbalizer, encoding="utf-8")
    claims = """# CH6 claim–evidence map\n\n| Claim | Domain | Evidence | Allowed wording | Forbidden wording |\n|---|---|---|---|---|\n| Public system route completed | ECG/brain | `result.json`, `audit.json`, provenance PNG | Framework constructed typed system evidence for this route. | The framework proved medical safety. |\n| Controlled integrity block | ECG/brain | control `audit.json` | A registered critical trace fault caused fail-closed block. | FuzzyXAI detected a medical error. |\n| Native XAI map | ECG/brain | IG/occlusion/Grad-CAM arrays | Map records local model sensitivity in its stated method space. | The highlighted anatomy/ECG segment caused the decision. |\n| IDRiD state | eyes | access note | Official data access remains missing. | The eye experiment was completed. |\n"""
    (REPORTS / "CH6_CLAIM_EVIDENCE_MAP.md").write_text(claims, encoding="utf-8")


def main() -> None:
    for directory in (REPORTS, TABLES, FIGURES): directory.mkdir(parents=True, exist_ok=True)
    ecg_run, ecg_cases, ecg_controls = ecg()
    brain_v1_runs = [(load(path), path.parent) for path in sorted((ROOT / "brain_allen" / "outputs" / "runs").glob("*/run.json"))]
    brain_v1_run, _ = min(brain_v1_runs, key=lambda item: (min(row["validation_loss"] for row in item[0]["history"]), item[0]["seed"]))
    brain_run, brain_cases, brain_controls = brain_v2()
    write_csv("T6_01_PUBLICATION_DATASET.csv", [{"domain":"eyes","relation_to_Averkin_work":"continuation; official access pending","dataset":"IDRiD","dataset_role":"primary","replication_or_continuation":"continuation"},{"domain":"ECG","relation_to_Averkin_work":"independent validation","dataset":"PTB-XL 1.0.3","dataset_role":"open validation","replication_or_continuation":"independent"},{"domain":"brain","relation_to_Averkin_work":"open atlas continuation","dataset":"Allen CCF 2017","dataset_role":"open continuation","replication_or_continuation":"continuation"}])
    ecg_prepared = Path(__import__("os").environ["FUZZYXAI_CH6_DATA_ROOT"]) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    ecg_labels, ecg_folds = np.load(ecg_prepared / "labels.npy"), np.load(ecg_prepared / "folds.npy")
    data_root = Path(__import__("os").environ["FUZZYXAI_CH6_DATA_ROOT"])
    brain_patches = load(data_root / "brain" / "allen_ccf_25um" / "prepared" / "patches.json")
    brain_v2_manifest = load(data_root / "brain" / "allen_ccf_25um" / "prepared_v2_confirmatory" / "dataset_manifest.json")
    write_csv("T6_02_DATASETS.csv", [
        {"domain":"eyes","N_total":"MISSING_DATA","N_train":"MISSING_DATA","N_val":"MISSING_DATA","N_test":"MISSING_DATA","modality":"fundus RGB","labels":"DR grading","license":"official authentication required"},
        {"domain":"ECG","N_total":len(ecg_labels),"N_train":int((ecg_folds<=8).sum()),"N_val":int((ecg_folds==9).sum()),"N_test":int((ecg_folds==10).sum()),"modality":"12-lead 100 Hz ECG","labels":"NORMAL / ABNORMAL","license":"Open Access; CC BY 4.0"},
        {"domain":"brain_v1_pilot","N_total":len(brain_patches),"N_train":sum(item["split"]=="train" for item in brain_patches),"N_val":sum(item["split"]=="validation" for item in brain_patches),"N_test":sum(item["split"]=="test" for item in brain_patches),"modality":"Nissl patch","labels":"OTHER gray matter / HPF","license":"Allen CCF public atlas"},
        {"domain":"brain_v2_confirmatory","N_total":sum(brain_v2_manifest["patch_counts"].values()),"N_train":brain_v2_manifest["patch_counts"]["train"],"N_val":brain_v2_manifest["patch_counts"]["validation"],"N_test":brain_v2_manifest["patch_counts"]["test"],"modality":"64×64 Nissl patch","labels":"OTHER gray matter / HPF","license":"Allen CCF public atlas"},
    ])
    write_csv("T6_03_MODELS.csv", [
        {"domain":"eyes","architecture":"VGG16/EfficientNetB0 scaffold","input_shape":"not executed","output":"MISSING_DATA","training_seeds":"not executed"},
        {"domain":"ECG","architecture":"ECGResNet1D","input_shape":"12×1000","output":"NORMAL/ABNORMAL","training_seeds":"2026, 2027, 2028"},
        {"domain":"brain","architecture":"InceptionV3","input_shape":"3×299×299","output":"OTHER/HPF","training_seeds":"2026, 2027, 2028"},
    ])
    write_csv("T6_04_MODEL_RESULTS.csv", [{"domain":"ECG", "seed": load(path)["seed"], **load(path)["test_metrics_calibrated"]} for path in sorted((ROOT / "ecg_ptbxl" / "outputs" / "runs").glob("*/run.json"))] + [{"domain":"brain_v1_pilot", "seed": load(path)["seed"], **load(path)["test_metrics_calibrated"]} for path in sorted((ROOT / "brain_allen" / "outputs" / "runs").glob("*/run.json"))] + [{"domain":"brain_v2_confirmatory", "seed": load(path)["seed"], **load(path)["test_metrics_calibrated"]} for path in sorted((ROOT / "brain_allen" / "outputs_v2_confirmatory" / "runs").glob("*/run.json"))])
    write_csv("T6_06_SELECTED_CASES.csv", ecg_cases + brain_cases)
    write_csv("T6_07_FUZZYXAI.csv", ecg_cases + brain_cases)
    write_csv("T6_08_CONTROLS.csv", ecg_controls + brain_controls)
    write_csv("T6_05_XAI.csv", [
        {"domain":"ECG","method":"Integrated Gradients","representation":"12×1000 signed tensor, fixed target logit","faithfulness":"top-10% masking vs random in per-case xai_diagnostics.json","spatial_diagnostic":"not applicable","limitations":"local sensitivity; not physiology causality"},
        {"domain":"ECG","method":"temporal occlusion","representation":"12×20 signed windows","faithfulness":"top-10% masking vs random","spatial_diagnostic":"12×20 common-grid diagnostic, not Gamma","limitations":"experiment-side secondary source"},
        {"domain":"brain_v2_confirmatory","method":"Grad-CAM + IG","representation":"299×299 map / full tensor","faithfulness":"not supplied","spatial_diagnostic":"HPF overlap only on positive patches; not Gamma","limitations":"single atlas, no causal claim"},
        {"domain":"eyes","method":"Grad-CAM + IG scaffold","representation":"not executed","faithfulness":"MISSING_DATA","spatial_diagnostic":"MISSING_DATA","limitations":"official IDRiD access required"},
    ])
    write_csv("T6_10_BEFORE_AFTER.csv", [{"property":key,"before_FuzzyXAI":before,"after_FuzzyXAI":after} for key,before,after in [("prediction","model output","unchanged"),("native XAI","optional isolated map","preserved as typed evidence"),("provenance","not unified","directed ExplanationGraph"),("alignment","not available","registered T_ij and Γ when plan applies"),("uncertainty","single model score","U_model/U_rules/U_trace/u_M status"),("system action","none","strict rho + critical override"),("missingness","implicit","required/optional/not_applicable"),("audit","ad hoc","public audit and reader report")]])
    strict_rows = slm_rows()
    write_csv("T6_11_CROSS_DOMAIN.csv", [{"domain":"eyes","data_status":"official access missing","model_status":"not executed","native_xai_status":"not executed","public_runtime":"not executed","slm_status":"not executed","limitations":"no fabricated output"},{"domain":"ECG","data_status":"executed","model_status":"executed","native_xai_status":"executed","public_runtime":"executed","slm_status":"executed strict" if any(row["domain"] == "ECG" for row in strict_rows) else "not executed","limitations":"technical, not clinical action"},{"domain":"brain_v1_pilot","data_status":"executed","model_status":"executed","native_xai_status":"pilot artifacts retained","public_runtime":"pilot only","slm_status":"not rerun","limitations":"single atlas and small held-out blocks"},{"domain":"brain_v2_confirmatory","data_status":"executed","model_status":"executed","native_xai_status":"executed","public_runtime":"executed","slm_status":"executed strict" if any(row["domain"] == "brain_v2_confirmatory" for row in strict_rows) else "not executed","limitations":"single atlas, section-block generalization"}])
    write_csv("T6_09_VERBALIZER.csv", strict_rows + [{"domain":"eyes","case":"not executed","P_fact":"MISSING_DATA","H":"MISSING_DATA","P_num":"MISSING_DATA","P_action":"MISSING_DATA","P_lim":"MISSING_DATA","status":"not executed"}])
    write_csv("T6_12_LIMITATIONS_AND_FAILURES.csv", [
        {"domain":"ECG","case":"ECG_D","what_happened":"confident false positive received technical accept","detected":"internally consistent route only","not_detected":"semantic prediction error","interpretation":"without external verification evidence, route consistency is not ground-truth correctness"},
        {"domain":"ECG","case":"ECG_E","what_happened":"confident false negative received technical accept","detected":"internally consistent route only","not_detected":"semantic prediction error","interpretation":"without external verification evidence, route consistency is not ground-truth correctness"},
        {"domain":"ECG","case":"ECG_I","what_happened":"controlled checkpoint mismatch","detected":"critical integrity fault","not_detected":"not applicable","interpretation":"critical override blocks the action"},
        {"domain":"ECG","case":"ECG_H","what_happened":"controlled provenance fault","detected":"critical integrity fault","not_detected":"not applicable","interpretation":"critical override blocks the action"},
        {"domain":"brain_v1_pilot","case":"all cases","what_happened":"single-atlas validation with seven held-out patches","detected":"limitation recorded in provenance/report","not_detected":"cross-atlas generalization","interpretation":"pilot metrics do not establish broad neuroanatomical transfer"},
        {"domain":"brain_v2_confirmatory","case":"all cases","what_happened":"single-atlas section-block validation","detected":"controlled integrity faults only","not_detected":"cross-atlas or clinical generalization","interpretation":"v2 increases anatomical sampling coverage but remains one-atlas evidence"},
    ])
    top_reports(ecg_cases, brain_cases, ecg_controls + brain_controls)
    save_json(ROOT / "reports" / "artifact_summary.json", {"ecg_canonical_run": ecg_run["run_id"], "brain_v1_pilot_canonical_run": brain_v1_run["run_id"], "brain_v2_confirmatory_canonical_run": brain_run["run_id"], "ecg_cases": ecg_cases, "brain_v2_cases": brain_cases, "strict_slm": strict_rows})
    print(ROOT)


if __name__ == "__main__": main()

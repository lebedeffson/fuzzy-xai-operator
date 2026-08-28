"""Project frozen PAPILA public artifacts into Chapter 6 tables and figures.

This is deliberately an exporter: system quantities are read from the stored
``ModelExplanationResult`` serialization and never recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_table(name: str, rows: list[dict[str, Any]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (TABLES / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_row(alias: str, selected: dict[str, Any], case_root: Path, native: dict[str, Any]) -> dict[str, Any]:
    cohort = "binary_outer_fold_5_test" if alias.startswith("EYE_") else "suspect_auxiliary"
    source = selected["cases"].get(alias) or selected["suspect_cases"][alias]
    sample = source["sample_id"]
    result = _load(case_root / sample / "result.json")
    system = result["system_evidence"]
    risk = system["risk"]
    uncertainty = system["uncertainty"]
    prediction = int(result["model"]["predictions"][0])
    probability = float(result["model"]["probabilities"][0][1])
    truth = source.get("ground_truth")
    return {
        "selection_id": alias, "physical_case_id": sample, "cohort": cohort,
        "truth": "N/A" if truth is None else truth, "prediction": prediction,
        "probability": probability, "correct": "N/A" if truth is None else prediction == int(truth),
        "native_xai_disagreement": native.get(sample),
        "native_xai_disagreement_scope": "registered positive-support L1 diagnostic; not canonical system Gamma",
        "system_Gamma": system["gamma"]["gamma"],
        "Gamma_scope": "canonical FuzzyXAI system alignment; not native-XAI-to-XAI agreement",
        "U_model": uncertainty["u_model"],
        "U_rules_status": uncertainty["sources"]["U_rules"]["status"],
        "U_trace": uncertainty["u_trace"], "u_M": uncertainty["u_m"],
        "Delta_status": system["reduction"]["status"],
        "Delta_display": "не применялось" if system["reduction"]["status"] == "not_applied" else system["reduction"].get("delta"),
        "I_pre": system["i_pre"]["value"], "rho": risk["rho"],
        "candidate_action": risk["candidate_action"], "critical_override": risk["critical_override"],
        "final_action": risk["action"],
        "missing_required": ",".join(risk.get("missing_required_components", [])),
        "optional_missing": ",".join(result.get("quality_status", {}).get("optional_missing", [])),
    }


def _native_diagnostics(data_root: Path) -> dict[str, float]:
    sweep = _load(data_root / "eyes" / "papila" / "xai_sweep_fold5.json")
    result: dict[str, float] = {}
    for row in sweep.get("rows", sweep if isinstance(sweep, list) else []):
        result[str(row["sample_id"])] = float(row["lime_gradcam_positive_support_l1_distance"])
    suspect = _load(data_root / "eyes" / "papila" / "xai_sweep_suspect.json")
    for row in suspect.get("rows", suspect if isinstance(suspect, list) else []):
        result[str(row["sample_id"])] = float(row["lime_gradcam_positive_support_l1_distance"])
    return result


def _controls_rows(controls: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control, item in controls.items():
        rows.append({"domain": "PAPILA", "control": control, **item})
    return rows


def _figures(rows: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    binary = [row for row in rows if row["cohort"] == "binary_outer_fold_5_test"]
    x = np.arange(len(binary))
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x - .25, [float(row["system_Gamma"]) for row in binary], .25, label="system Γ")
    axis.bar(x, [float(row["u_M"]) for row in binary], .25, label="u_M")
    axis.bar(x + .25, [float(row["rho"]) for row in binary], .25, label="ρ")
    axis.set(xticks=x, xticklabels=[row["selection_id"] for row in binary], title="PAPILA: selected public FuzzyXAI cases")
    axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "PAPILA_SYSTEM_METRICS.png", dpi=180); plt.close(fig)
    keys = list(controls)
    fig, axis = plt.subplots(figsize=(10, 4))
    colors = ["tab:red" if controls[key]["critical_override"] else "tab:blue" for key in keys]
    axis.bar(np.arange(len(keys)), [float(controls[key]["rho"] or 0.0) for key in keys], color=colors)
    axis.axhline(.35, color="black", linestyle="--", linewidth=1, label="theta1")
    axis.set(xticks=np.arange(len(keys)), xticklabels=keys, ylabel="ρ", title="PAPILA controlled route-integrity matrix")
    axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "PAPILA_MODEL_ERROR_VS_INTEGRITY_FAULT.png", dpi=180); plt.close(fig)


def _diagnostic_exports() -> None:
    """Render already measured diagnostics; no map or risk value is recomputed."""
    for stem, title in (("papila_spatial_diagnostics", "PAPILA spatial correspondence diagnostics"),
                        ("papila_annotation_variability", "PAPILA expert annotation variability"),
                        ("papila_faithfulness", "PAPILA perturbation faithfulness")):
        source = TABLES / f"{stem}.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        (TABLES / f"{stem}.md").write_text(
            f"# {title}\n\nSource table is a registered diagnostic only; it is neither canonical system Gamma nor causal validation.\n\n"
            + "| " + " | ".join(rows[0]) + " |\n| " + " | ".join("---" for _ in rows[0]) + " |\n"
            + "\n".join("| " + " | ".join(str(row[key]) for key in rows[0]) + " |" for row in rows) + "\n",
            encoding="utf-8",
        )
    faith = []
    with (TABLES / "papila_faithfulness.csv").open(encoding="utf-8", newline="") as stream:
        faith = list(csv.DictReader(stream))
    fig, axis = plt.subplots(figsize=(10, 4))
    labels = [f"{row['selection_id']}\n{row['method'].replace('_support', '')}" for row in faith]
    axis.bar(np.arange(len(faith)), [float(row["xai_minus_random"]) for row in faith], color=["tab:green" if float(row["xai_minus_random"]) > 0 else "tab:orange" for row in faith])
    axis.axhline(0, color="black", linewidth=1); axis.set(xticks=np.arange(len(faith)), xticklabels=labels, ylabel="XAI drop − random drop", title="PAPILA: perturbation faithfulness is mixed")
    fig.tight_layout(); fig.savefig(FIGURES / "PAPILA_FAITHFULNESS.png", dpi=180); plt.close(fig)


def _aggregate_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load(path)
    output: list[dict[str, Any]] = []
    for cohort in ("binary_outer_fold_5_test", "suspect_auxiliary"):
        rows = [row for row in payload["rows"] if row["cohort"] == cohort]
        actions = {action: sum(row["final_action"] == action for row in rows) for action in sorted({row["final_action"] for row in rows})}
        numeric = ("system_Gamma", "U_model", "U_trace", "u_M", "I_pre", "rho")
        for key in numeric:
            values = np.asarray([float(row[key]) for row in rows if row[key] is not None], dtype=float)
            output.append({"cohort": cohort, "quantity": key, "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1) if len(values) > 1 else 0.0, "median": np.median(values), "q1": np.quantile(values, .25), "q3": np.quantile(values, .75), "min": values.min(), "max": values.max(), "complete": sum(row["risk_status"] == "complete" for row in rows), "incomplete": sum(row["risk_status"] != "complete" for row in rows), "critical": sum(bool(row["critical_override"]) for row in rows), "action_distribution": json.dumps(actions, ensure_ascii=False)})
    return output


def _update_reports(rows: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    eye = REPORTS / "CH6_EYE_RESULTS_RU.md"
    eye_text = (
        "# Глава 6 — офтальмология: PAPILA\n\n"
        "## Данные и frozen protocol\n\n"
        "Офтальмологическая empirical line выполнена на официальном PAPILA v2: 244 пациента, 488 RGB fundus images; healthy=333, glaucoma=87, suspect=68. Primary binary protocol исключает целиком suspect-associated patients: clean binary cohort=210 patients, auxiliary suspect cohort=34 patients. ResNet50, patient-level five-fold split (seed 2026) и canonical outer fold 5 / seed 2026 заморожены до native-XAI анализа. Fixed-seed CV: AUROC=0.6874±0.0637, balanced accuracy=0.5990±0.0718. Это умеренный, а не скрытый «красивый» результат; FuzzyXAI не улучшает classifier performance.\n\n"
        "## Native XAI и diagnostics\n\n"
        "LIME (SLIC, 50 superpixels, 1000 perturbations, seed 2026) и predicted-class Grad-CAM (`layer4.2.conv3`) сохранены как разные native evidence. Registered positive-support L1 discrepancy — отдельный native-XAI diagnostic, а не `system_Gamma`; EYE_F=RET135OD имеет валидный frozen maximum 1.0 при finite/non-zero maps. Disc/cup/rim energy, pointing and top-support overlaps — spatial localization diagnostics, не causal ground truth. Expert1↔Expert2 Dice/CDR отражает вариативность предметной аннотации, а не ошибку эксперта. Faithfulness uses a frozen 10% support mask and 20 random equal-area controls. Its mixed/negative results are retained: spatial localization and perturbation faithfulness are not interchangeable criteria, and no map is therefore called universally faithful.\n\n"
        "## Public FuzzyXAI и controls\n\n"
        f"Full public `FuzzyXAI.wrap(...).explain_one(...)` artifacts exist for {len(rows)} selected aliases (physical duplicates are explicitly mapped). `system_Gamma` is canonical alignment from the registered probability-to-technical-risk interface, not LIME↔Grad-CAM agreement. Reduction is not supported by the frozen plan, so chapter-facing Δ is ‘не применялось’. Controls are factual controlled integrity injections; numeric rho is retained, while a critical registered rupture fail-closes final action. The frozen plan declares Grad-CAM required, therefore CONTROL_1 is documented as a missing-required source rather than retrospectively relabelled optional.\n\n"
        "## Boundary of system explainability\n\n"
        "EYE_D (false positive) and EYE_E (false negative) are kept as model-error cases. A route may be internally consistent and receive a non-blocking technical action while its prediction is wrong: FuzzyXAI controls registered evidence integrity, uncertainty and provenance, not external diagnostic truth without a reference-verification channel. Conversely checkpoint, target and patient-linkage control faults test the fail-closed route policy. The suspect cohort is only descriptive behavior of the frozen binary model on excluded clinically ambiguous cases; no accuracy, FP/FN, sensitivity or specificity is assigned to it.\n\n"
        "## Limitations\n\n"
        "PAPILA has a small glaucoma class, one public dataset and expert ROI for a controlled experiment; it is not multicenter clinical validation. LIME is a local surrogate and Grad-CAM has coarse spatial resolution. Overlap with disc/cup structures is not causal validation. Technical FuzzyXAI actions are not clinical decisions.\n"
    )
    eye.write_text(eye_text, encoding="utf-8")
    (REPORTS / "CH6_EYES_RESULTS_RU.md").write_text(eye_text, encoding="utf-8")
    (REPORTS / "CH6_CROSS_DOMAIN_RESULTS_RU.md").write_text(
        "# Глава 6 — три эмпирических домена\n\n"
        "Выполнены три public-runtime validation lines: PAPILA fundus RGB / ResNet50 / LIME+Grad-CAM; PTB-XL 12-lead ECG / ECGResNet1D / IG+temporal occlusion; Allen CCF 2017 Nissl / InceptionV3 / Grad-CAM+IG. FuzzyXAI core unchanged; domain-specific only are factual data and model adapters, native XAI, registered transforms, ExplainPlan and vocabulary. PAPILA deliberately has moderate classifier performance, ECG is strong but imperfect, and Allen is single-atlas; this contrast shows the framework contract is not conditional on a near-perfect base classifier. Native-XAI disagreement, spatial diagnostics and canonical system Γ are separate quantities.\n",
        encoding="utf-8",
    )
    (REPORTS / "CH6_MEDICAL_RESULTS_RU.md").write_text(
        "# Глава 6 — medical validation\n\n"
        "The completed empirical chapter comprises PAPILA glaucoma fundus images, PTB-XL ECG and Allen CCF neuroanatomical images. Each route uses public `FuzzyXAI.wrap(...).explain_one(...)` and produces typed evidence, system evidence, reader/audit projections and provenance. System Γ denotes registered system-interface alignment; it is not agreement of two native XAI maps. Controlled integrity faults can cause fail-closed block, whereas a confident internally consistent model mistake may remain non-blocking without external truth evidence. This distinction is a stated limitation, not a claim that FuzzyXAI detects all model errors.\n",
        encoding="utf-8",
    )


def _slm_rows(case_root: Path) -> list[dict[str, Any]]:
    aliases = {
        "EYE_A": "RET038OS", "EYE_B": "RET098OS", "EYE_C": "RET119OS", "EYE_D": "RET170OD",
        "EYE_E": "RET265OS", "EYE_F": "RET135OD", "SUSPECT_A": "RET009OD", "SUSPECT_D": "RET009OD",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    examples: list[str] = []
    for alias, sample in aliases.items():
        payload = _load(case_root / sample / "strict_slm.json")
        rows.append({"selection_id": alias, "physical_case_id": sample, "unique_physical_result": sample not in seen, "status": payload["status"], **payload["metrics"], "model": payload["model_id"], "revision": payload["revision"], "prompt_profile_sha256": payload["prompt_profile_sha256"]})
        if sample not in seen and alias in {"EYE_A", "EYE_B"}:
            technical = _load(case_root / sample / "result.json")["human_explanations"]["domain_user"]
            examples.append(f"## {alias} / {sample}\n\nТехнический certified text:\n\n{technical['decision']['explanation']}\n\nStrict ophthalmology text:\n\n{payload['text']}\n")
        seen.add(sample)
    (REPORTS / "CH6_VERBALIZER_RESULTS_RU.md").write_text(
        "# Глава 6 — strict SLM verbalizer\n\n"
        "For PAPILA the same pinned local strict backend as ECG and Allen was used: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. It receives only certified public HumanExplanation claims, never raw image or model internals. H=0 for every generated accepted output; rejected/fallback status would be retained instead of hidden. The table preserves P_fact, H, P_num, P_action and P_lim rather than judging literary quality.\n\n"
        + "\n".join(examples), encoding="utf-8",
    )
    return rows


def _global_tables(rows: list[dict[str, Any]], slm: list[dict[str, Any]]) -> None:
    """Replace stale two-domain chapter projections with the executed triad."""
    _write_table("T6_02_DATASETS", [
        {"domain": "PAPILA", "dataset": "PAPILA v2 / Figshare 14798004", "modality": "RGB fundus", "status": "executed", "scope": "patient-level healthy/glaucoma; separate suspect auxiliary cohort"},
        {"domain": "PTB-XL", "dataset": "PTB-XL 1.0.3", "modality": "12-lead ECG", "status": "executed", "scope": "registered official folds"},
        {"domain": "Allen CCF", "dataset": "Allen CCF 2017", "modality": "Nissl atlas", "status": "executed", "scope": "single-atlas section-block validation"},
    ])
    _write_table("T6_03_MODELS", [
        {"domain": "PAPILA", "model": "ResNet50", "canonical": "outer fold 5 / seed 2026", "selection": "minimum validation loss"},
        {"domain": "PTB-XL", "model": "ECGResNet1D", "canonical": "registered run", "selection": "validation-only"},
        {"domain": "Allen CCF", "model": "InceptionV3", "canonical": "brain_v2_confirmatory", "selection": "validation-only"},
    ])
    _write_table("T6_04_MODEL_RESULTS", [
        {"domain": "PAPILA", "AUROC_mean_sd": "0.6874 ± 0.0637", "balanced_accuracy_mean_sd": "0.5990 ± 0.0718", "interpretation": "moderate frozen classifier performance"},
        {"domain": "PTB-XL", "AUROC_mean_sd": "see registered canonical run", "balanced_accuracy_mean_sd": "see registered canonical run", "interpretation": "strong but imperfect classifier"},
        {"domain": "Allen CCF", "AUROC_mean_sd": "see brain_v2_confirmatory", "balanced_accuracy_mean_sd": "see brain_v2_confirmatory", "interpretation": "single-atlas task"},
    ])
    _write_table("T6_05_XAI", [
        {"domain": "PAPILA", "native_xai": "LIME + Grad-CAM", "native_diagnostic": "positive-support L1; not system Gamma"},
        {"domain": "PTB-XL", "native_xai": "Integrated Gradients + temporal occlusion", "native_diagnostic": "IG/occlusion; not system Gamma"},
        {"domain": "Allen CCF", "native_xai": "Grad-CAM + Integrated Gradients", "native_diagnostic": "GradCAM/IG; not system Gamma"},
    ])
    _write_table("T6_09_VERBALIZER", slm)
    _write_table("T6_10_BEFORE_AFTER", [
        {"domain": "PAPILA", "before": "prediction, P(glaucoma), LIME, Grad-CAM", "after": "same prediction, provenance, system Gamma, uncertainty, I_pre, rho/action, strict verbalization"},
        {"domain": "PTB-XL", "before": "prediction and native maps", "after": "public system evidence and audit"},
        {"domain": "Allen CCF", "before": "prediction and native maps", "after": "public system evidence and audit"},
    ])
    _write_table("T6_11_CROSS_DOMAIN", [
        {"domain": "PAPILA", "input": "RGB fundus", "model": "ResNet50", "native_xai": "LIME + Grad-CAM", "public_route": "executed"},
        {"domain": "PTB-XL", "input": "12×1000 temporal signal", "model": "ECGResNet1D", "native_xai": "IG + temporal occlusion", "public_route": "executed"},
        {"domain": "Allen CCF", "input": "Nissl image", "model": "InceptionV3", "native_xai": "Grad-CAM + IG", "public_route": "executed"},
    ])
    _write_table("T6_12_LIMITATIONS_AND_FAILURES", [
        {"domain": "PAPILA", "case": "EYE_D/EYE_E", "what_happened": "confident model error can retain a consistent technical route", "detected": "registered route integrity only", "not_detected": "external diagnostic truth"},
        {"domain": "PAPILA", "case": "controls", "what_happened": "checkpoint/target/linkage injected faults", "detected": "critical override -> block", "not_detected": "clinical validity"},
        {"domain": "PAPILA", "case": "faithfulness", "what_happened": "mixed/negative perturbation results", "detected": "diagnostic contrast to random masking", "not_detected": "causality"},
        {"domain": "Allen CCF", "case": "brain_v2", "what_happened": "single-atlas task", "detected": "section-block provenance", "not_detected": "cross-atlas generalization"},
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args()
    selected = _load(args.data_root / "eyes" / "papila" / "selected_cases_eye_v3.json")
    native = _native_diagnostics(args.data_root)
    aliases = [f"EYE_{letter}" for letter in "ABCDEFGH"] + [f"SUSPECT_{letter}" for letter in "ABCDE"]
    rows = [_case_row(alias, selected, args.case_root, native) for alias in aliases]
    controls = _load(args.controls)
    _write_table("T6_EYE_FUZZYXAI", rows)
    _write_table("T6_EYE_CONTROLS", _controls_rows(controls))
    _write_table("T6_06_SELECTED_CASES", rows)
    _write_table("T6_07_FUZZYXAI", rows)
    _write_table("T6_08_CONTROLS", _controls_rows(controls))
    slm = _slm_rows(args.case_root)
    _write_table("T6_EYE_VERBALIZER", slm)
    _global_tables(rows, slm)
    _figures(rows, controls); _diagnostic_exports(); _update_reports(rows, controls)
    aggregate_path = args.data_root / "eyes" / "papila" / "papila_lightweight_public_route_v1.json"
    if aggregate_path.exists():
        _write_table("T6_EYE_AGGREGATE_ROUTE", _aggregate_rows(aggregate_path))
    print(json.dumps({"selected_aliases": len(rows), "unique_physical_cases": len({row['physical_case_id'] for row in rows})}, ensure_ascii=False))


if __name__ == "__main__":
    main()

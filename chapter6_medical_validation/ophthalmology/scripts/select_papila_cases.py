"""Programmatic PAPILA case selection from frozen canonical artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def _quality(path: Path) -> dict[str, float | bool]:
    with Image.open(path) as image: rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()); low=float((gray < 20).mean()); high=float((gray > 235).mean()); coverage=float((gray > 8).mean())
    # Frozen monotone technical score; thresholds are not selected on test labels.
    score = min(1.0, blur / 100.0) * (1.0-low) * (1.0-high) * coverage
    return {"blur_score": blur, "underexposure_fraction": low, "overexposure_fraction": high, "field_of_view_coverage": coverage, "finite_valid": bool(np.isfinite(rgb).all()), "technical_quality_score": float(score)}


def _cup_dice(contours: Path, sample_id: str) -> dict[str, float | None]:
    paths = [contours / f"{sample_id}_cup_exp{item}.txt" for item in (1, 2)]
    if not all(item.is_file() for item in paths): return {"cup_dice": None, "cdr_expert1": None, "cdr_expert2": None, "cdr_absolute_difference": None}
    points = [np.loadtxt(item, dtype=np.float32) for item in paths]; canvas = np.zeros((1934, 2576), dtype=np.uint8); masks=[]
    for value in points:
        mask=canvas.copy(); cv2.fillPoly(mask,[np.round(value).astype(np.int32)],1); masks.append(mask.astype(bool))
    union=(masks[0]|masks[1]).sum(); dice=2*(masks[0]&masks[1]).sum()/(masks[0].sum()+masks[1].sum()) if masks[0].sum()+masks[1].sum() else 0.0
    disc_paths=[contours / f"{sample_id}_disc_exp{item}.txt" for item in (1,2)]; cdr=[]
    for cup,disc_path in zip(masks,disc_paths,strict=True):
        disc=np.zeros_like(canvas); cv2.fillPoly(disc,[np.round(np.loadtxt(disc_path,dtype=np.float32)).astype(np.int32)],1); cdr.append(float(np.sqrt(cup.sum()/max(disc.sum(),1))))
    return {"cup_dice": float(dice), "cdr_expert1": cdr[0], "cdr_expert2": cdr[1], "cdr_absolute_difference": abs(cdr[0]-cdr[1])}


def _entry(row: dict[str, Any], case_id: str, rule: str, checkpoint: str, refs: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value={"case_id":case_id,"selection_rule":rule,"sample_id":row["sample_id"],"patient_id":row["sample_id"][:-2],"eye":row["sample_id"][-2:],"ground_truth":int(row["label"]),"prediction":int(np.argmax(row["probabilities"])),"p_glaucoma":float(row["probabilities"][1]),"correct":bool(int(np.argmax(row["probabilities"]))==int(row["label"])),"checkpoint_sha256":checkpoint,"source_refs":refs}
    if extra: value.update(extra)
    return value


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--data-root",type=Path,required=True); parser.add_argument("--canonical-run",type=Path,required=True); parser.add_argument("--suspect-predictions",type=Path); parser.add_argument("--fold5-xai-sweep",type=Path); parser.add_argument("--suspect-xai-sweep",type=Path); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    run=json.loads((args.canonical_run/"run.json").read_text()); pred=json.loads((args.canonical_run/"test_predictions.json").read_text())["rows"]; labels={row["sample_id"]:row for row in _rows(args.data_root/"eyes"/"papila"/"verified"/"papila_eye_labels.csv")}; raw=next((args.data_root/"eyes"/"papila"/"raw").glob("PapilaDB-PAPILA-*")); contours=raw/"ExpertsSegmentations"/"Contours"
    for row in pred:
        source=labels[row["sample_id"]]; row.update({"image_path":str(args.data_root/source["image_path"]),"image_sha256":source["image_sha256"]}); row.update(_quality(Path(row["image_path"]))); row.update(_cup_dice(contours,row["sample_id"]))
    confidence=lambda row:max(row["probabilities"]); predicted=lambda row:int(np.argmax(row["probabilities"])); base=[str(args.canonical_run/"run.json"),str(args.data_root/"eyes"/"papila"/"verified"/"papila_cv_folds_seed2026.json")]
    correct_h=[r for r in pred if r["label"]==0 and predicted(r)==0]; correct_g=[r for r in pred if r["label"]==1 and predicted(r)==1]; fp=[r for r in pred if r["label"]==0 and predicted(r)==1]; fn=[r for r in pred if r["label"]==1 and predicted(r)==0]
    selected={"EYE_A":_entry(max(correct_h,key=confidence),"EYE_A","highest-confidence correct HEALTHY",run["checkpoint_sha256"],base),"EYE_B":_entry(max(correct_g,key=confidence),"EYE_B","highest-confidence correct GLAUCOMA",run["checkpoint_sha256"],base),"EYE_C":_entry(min(pred,key=lambda r:abs(r["probabilities"][1]-0.5)),"EYE_C","minimum |P(glaucoma)-0.5|",run["checkpoint_sha256"],base),"EYE_D":_entry(max(fp,key=confidence),"EYE_D","highest-confidence false positive",run["checkpoint_sha256"],base) if fp else {"case_id":"EYE_D","status":"not_available","selection_rule":"highest-confidence false positive"},"EYE_E":_entry(max(fn,key=confidence),"EYE_E","highest-confidence false negative",run["checkpoint_sha256"],base) if fn else {"case_id":"EYE_E","status":"not_available","selection_rule":"highest-confidence false negative"},"EYE_G":_entry(min(pred,key=lambda r:r["technical_quality_score"]),"EYE_G","lowest frozen technical image-quality score",run["checkpoint_sha256"],base),"EYE_H":_entry(max((r for r in pred if r["cup_dice"] is not None),key=lambda r:1-r["cup_dice"]),"EYE_H","maximum expert1/expert2 optic-cup disagreement (1-Dice)",run["checkpoint_sha256"],base)}
    selected["EYE_F"]={"case_id":"EYE_F","status":"pending_registered_lime_gradcam_diagnostic","selection_rule":"maximum registered LIME/Grad-CAM disagreement over canonical test set","checkpoint_sha256":run["checkpoint_sha256"],"source_refs":base}
    if args.fold5_xai_sweep:
        value=max(json.loads(args.fold5_xai_sweep.read_text(encoding="utf-8"))["rows"],key=lambda item:item["lime_gradcam_positive_support_l1_distance"])
        selected["EYE_F"]={"case_id":"EYE_F","selection_rule":"maximum registered LIME/Grad-CAM positive-support L1 diagnostic","sample_id":value["sample_id"],"patient_id":value["patient_id"],"eye":value["eye"],"ground_truth":value["source_diagnosis"],"prediction":value["prediction"],"p_glaucoma":value["p_glaucoma"],"lime_gradcam_positive_support_l1_distance":value["lime_gradcam_positive_support_l1_distance"],"checkpoint_sha256":run["checkpoint_sha256"],"source_refs":[*base,str(args.fold5_xai_sweep)]}
    suspect_cases: dict[str, Any] = {}
    if args.suspect_predictions:
        suspect=json.loads(args.suspect_predictions.read_text(encoding="utf-8"))["rows"]
        def suspect_entry(row: dict[str, Any], name: str, rule: str) -> dict[str, Any]:
            return {"case_id":name,"cohort":"PAPILA_SUSPECT_AUXILIARY_COHORT","selection_rule":rule,"sample_id":row["sample_id"],"patient_id":row["patient_id"],"eye":row["eye"],"source_diagnosis":row["source_diagnosis"],"p_glaucoma":row["p_glaucoma"],"binary_prediction":int(row["p_glaucoma"]>=0.5),"u_model":row["u_model_entropy"],"checkpoint_sha256":run["checkpoint_sha256"],"semantics":"binary-model behavior; no suspect accuracy or FP/FN claim"}
        suspect_cases={"SUSPECT_A":suspect_entry(min(suspect,key=lambda r:abs(r["p_glaucoma"]-.5)),"SUSPECT_A","P(glaucoma) nearest 0.5"),"SUSPECT_B":suspect_entry(max(suspect,key=lambda r:r["p_glaucoma"]),"SUSPECT_B","highest P(glaucoma)"),"SUSPECT_C":suspect_entry(min(suspect,key=lambda r:r["p_glaucoma"]),"SUSPECT_C","lowest P(glaucoma)"),"SUSPECT_D":suspect_entry(max(suspect,key=lambda r:r["u_model_entropy"]),"SUSPECT_D","highest binary entropy U_model"),"SUSPECT_E":{"case_id":"SUSPECT_E","status":"pending_registered_lime_gradcam_diagnostic","selection_rule":"highest registered LIME/Grad-CAM disagreement over suspect auxiliary cohort","checkpoint_sha256":run["checkpoint_sha256"]}}
        if args.suspect_xai_sweep:
            value=max(json.loads(args.suspect_xai_sweep.read_text(encoding="utf-8"))["rows"],key=lambda item:item["lime_gradcam_positive_support_l1_distance"])
            suspect_cases["SUSPECT_E"]={"case_id":"SUSPECT_E","cohort":"PAPILA_SUSPECT_AUXILIARY_COHORT","selection_rule":"maximum registered LIME/Grad-CAM positive-support L1 diagnostic","sample_id":value["sample_id"],"patient_id":value["patient_id"],"eye":value["eye"],"source_diagnosis":value["source_diagnosis"],"binary_prediction":value["prediction"],"p_glaucoma":value["p_glaucoma"],"lime_gradcam_positive_support_l1_distance":value["lime_gradcam_positive_support_l1_distance"],"checkpoint_sha256":run["checkpoint_sha256"],"semantics":"binary-model behavior; no suspect accuracy or FP/FN claim","source_refs":[str(args.suspect_xai_sweep)]}
    payload={"schema_version":"1.0","cohort":"canonical_outer_fold_5_binary_test","canonical_run_id":run["run_id"],"canonical_seed":run["seed"],"selection_criterion":"minimum validation loss among registered fold-5 seeds; never outer-test performance","checkpoint_sha256":run["checkpoint_sha256"],"cases":selected,"suspect_cases":suspect_cases,"quality_definition":"min(1, Laplacian_variance/100)*(1-underexposure)*(1-overexposure)*field_of_view_coverage","note":"EYE_F and SUSPECT_E are intentionally pending until their registered two-native-XAI diagnostic has been computed over each eligible cohort."}; payload["manifest_sha256"]=sha256_json(payload); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(args.output)

if __name__=="__main__": main()

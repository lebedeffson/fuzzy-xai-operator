"""Cohort-wide native-XAI diagnostic used only to select F/E cases.

It deliberately exports no ModelExplanationResult and never computes system
Gamma/risk.  The reported distance is a registered LIME-positive vs Grad-CAM
spatial diagnostic, not canonical FuzzyXAI Gamma.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np, torch
from chapter6_medical_validation.ophthalmology.src.datasets import load_yaml
from chapter6_medical_validation.ophthalmology.src.lime_image import explain_lime
from chapter6_medical_validation.ophthalmology.src.models import build_classifier, resolve_module
from chapter6_medical_validation.ophthalmology.src.native_xai import grad_cam
from chapter6_medical_validation.ophthalmology.src.papila import expert1_disc_roi, papila_tensor
from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file

ROOT=Path(__file__).resolve().parents[1]
def _rows(path:Path):
 with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def _norm32(values:np.ndarray)->np.ndarray:
 value=cv2.resize(np.asarray(values,dtype=np.float64),(32,32),interpolation=cv2.INTER_AREA); value=np.maximum(value,0); total=value.sum();return value/total if total else value
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--canonical-run',type=Path,required=True);p.add_argument('--cohort',choices=['fold5','suspect'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=json.loads((a.canonical_run/'run.json').read_text());root=a.data_root;labels={r['sample_id']:r for r in _rows(root/'eyes/papila/verified/papila_eye_labels.csv')}; split=json.loads((root/'eyes/papila/verified/papila_cv_folds_seed2026.json').read_text()); test=set(split['folds']['5']['test_patient_ids']); suspect=set(split['suspect_auxiliary_patients']); chosen=test if a.cohort=='fold5' else suspect; rows=[r for r in labels.values() if r['patient_id'] in chosen];cfg=load_yaml(ROOT/'configs/preprocessing_papila.yaml');raw=next((root/'eyes/papila/raw').glob('PapilaDB-PAPILA-*'));contours=raw/'ExpertsSegmentations/Contours';model=build_classifier('resnet50',num_classes=2,pretrained=True).cuda().eval();model.load_state_dict(torch.load(a.canonical_run/'best_model.pt',map_location='cuda',weights_only=False)['state_dict']);checkpoint=sha256_file(a.canonical_run/'best_model.pt');result=[]
 for ordinal,row in enumerate(rows,1):
  image=root/row['image_path'];contour=contours/f"{row['sample_id']}_disc_exp1.txt";display=cv2.resize(expert1_disc_roi(image,contour,margin_fraction=float(cfg['roi_margin_fraction'])),(224,224),interpolation=cv2.INTER_AREA); tensor=papila_tensor(image,contour,cfg,training=False,seed=None)[None]
  def transform(value:object):
   data=np.asarray(value,dtype=np.float32).reshape(224,224,3);mean=np.asarray(cfg['mean'],dtype=np.float32);std=np.asarray(cfg['std'],dtype=np.float32);return torch.from_numpy(((data/255-mean)/std).transpose(2,0,1).astype(np.float32))[None].cuda()
  def predict(images:np.ndarray):
   with torch.no_grad():
    batch=np.stack([transform(item)[0].cpu().numpy() for item in images]);return torch.softmax(model(torch.from_numpy(batch).cuda()),dim=1).cpu().numpy()
  probs=predict(display[None])[0];target=int(np.argmax(probs));lime=explain_lime(display,predict,target_class=target);cam=grad_cam(model,torch.from_numpy(tensor).cuda(),resolve_module(model,'layer4.2.conv3'),target_layer_id='layer4.2.conv3',sample_id=row['sample_id'],checkpoint_sha256=checkpoint,target_class=target);l,g=_norm32(lime.positive_map),_norm32(cam.normalized_map);distance=float(np.abs(l-g).sum()/2);result.append({'sample_id':row['sample_id'],'patient_id':row['patient_id'],'eye':row['eye'],'source_diagnosis':int(row['diagnosis']),'prediction':target,'p_glaucoma':float(probs[1]),'lime_gradcam_positive_support_l1_distance':distance,'lime_target':lime.target_class,'gradcam_target':cam.target_class,'lime_parameters':{'segments':50,'perturbations':1000,'seed':2026,'kernel_width':0.25},'transform_lime':{'id':'T_LIME_positive_support_32_l1_v1','negative_coefficients_preserved_separately':True},'transform_gradcam':{'id':'T_GRADCAM_positive_support_32_l1_v1'},'checkpoint_sha256':checkpoint})
  print(f'{a.cohort} {ordinal}/{len(rows)} {row["sample_id"]}',flush=True)
 payload={'schema_version':'1.0','cohort':a.cohort,'semantics':'registered native-XAI positive-support diagnostic; not canonical system Gamma','canonical_run_id':run['run_id'],'checkpoint_sha256':checkpoint,'rows':result};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(a.output)
if __name__=='__main__':main()

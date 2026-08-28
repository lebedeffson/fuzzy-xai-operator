"""Frozen-model predictions for the suspect-associated ambiguity cohort only."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from chapter6_medical_validation.ophthalmology.src.datasets import load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier
from chapter6_medical_validation.ophthalmology.src.papila import papila_tensor

ROOT=Path(__file__).resolve().parents[1]
def rows(path:Path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--canonical-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=json.loads((a.canonical_run/'run.json').read_text());split=json.loads((a.data_root/'eyes/papila/verified/papila_cv_folds_seed2026.json').read_text());labels=rows(a.data_root/'eyes/papila/verified/papila_eye_labels.csv');sus=set(split['suspect_auxiliary_patients']);cfg=load_yaml(ROOT/'configs/preprocessing_papila.yaml');raw=next((a.data_root/'eyes/papila/raw').glob('PapilaDB-PAPILA-*'));contours=raw/'ExpertsSegmentations/Contours';model=build_classifier('resnet50',num_classes=2,pretrained=True).cuda().eval();model.load_state_dict(torch.load(a.canonical_run/'best_model.pt',map_location='cuda',weights_only=False)['state_dict']);out=[]
 with torch.no_grad():
  for row in labels:
   if row['patient_id'] not in sus:continue
   tensor=papila_tensor(a.data_root/row['image_path'],contours/f"{row['sample_id']}_disc_exp1.txt",cfg,training=False,seed=None)[None];prob=torch.softmax(model(torch.from_numpy(tensor).cuda()),dim=1)[0].cpu().numpy();entropy=float(-np.sum(prob*np.log(np.maximum(prob,1e-12)))/np.log(2));out.append({'sample_id':row['sample_id'],'patient_id':row['patient_id'],'eye':row['eye'],'source_diagnosis':int(row['diagnosis']),'p_glaucoma':float(prob[1]),'probabilities':prob.tolist(),'u_model_entropy':entropy,'checkpoint_sha256':run['checkpoint_sha256']})
 payload={'schema_version':'1.0','cohort':'PAPILA_SUSPECT_AUXILIARY_COHORT','semantics':'frozen binary-model behavior on clinically ambiguous excluded patients; not diagnostic accuracy','canonical_run_id':run['run_id'],'checkpoint_sha256':run['checkpoint_sha256'],'rows':out};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(a.output)
if __name__=='__main__':main()

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from chapter6_medical_validation.ophthalmology.src.datasets import load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier
from chapter6_medical_validation.ophthalmology.src.papila import expert1_disc_roi
import cv2
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.cases.read_text());cfg=load_yaml(ROOT/'configs/preprocessing_papila.yaml');raw=next((a.data_root/'eyes/papila/raw').glob('PapilaDB-PAPILA-*'));cont=raw/'ExpertsSegmentations/Contours';model=build_classifier('resnet50',num_classes=2,pretrained=True).cuda().eval();model.load_state_dict(torch.load(a.run/'best_model.pt',map_location='cuda',weights_only=False)['state_dict']);out=[]
 for alias in ('EYE_A','EYE_B','EYE_C','EYE_D','EYE_E','EYE_F'):
  sid=s['cases'][alias]['sample_id'];image=cv2.resize(expert1_disc_roi(raw/'FundusImages'/f'{sid}.jpg',cont/f'{sid}_disc_exp1.txt',margin_fraction=float(cfg['roi_margin_fraction'])),(224,224));
  def predict(batch):
   x=np.asarray(batch,np.float32)/255; x=((x-np.asarray(cfg['mean']))/np.asarray(cfg['std'])).transpose(0,3,1,2);return torch.softmax(model(torch.from_numpy(x.astype(np.float32)).cuda()),dim=1).detach().cpu().numpy()
  prob=predict(image[None])[0];target=int(prob.argmax());rng=np.random.default_rng(2026);n=int(.1*224*224)
  for method,file in [('lime_positive_support','lime_positive_map.npy'),('grad_cam','grad_cam_raw.npy')]:
   h=np.load(a.data_root/'eyes/papila/cases_v2'/sid/file);top=np.argpartition(h.ravel(),-n)[-n:];base=image.mean(axis=(0,1));m=image.copy().reshape(-1,3);m[top]=base;drop=float(prob[target]-predict(m.reshape(image.shape)[None])[0,target]);rd=[]
   for _ in range(20):
    m=image.copy().reshape(-1,3);m[rng.choice(h.size,n,replace=False)]=base;rd.append(float(prob[target]-predict(m.reshape(image.shape)[None])[0,target]))
   out.append({'selection_id':alias,'physical_case_id':sid,'method':method,'target':target,'area_fraction':.1,'random_repeats':20,'p_original':float(prob[target]),'xai_drop':drop,'random_drop_mean':float(np.mean(rd)),'random_drop_std':float(np.std(rd,ddof=1)),'xai_minus_random':float(drop-np.mean(rd)),'semantics':'perturbation faithfulness diagnostic; not causal validation'})
 a.output.parent.mkdir(parents=True,exist_ok=True);import csv
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(a.output)
if __name__=='__main__':main()

from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from chapter6_medical_validation.ophthalmology.src.papila import contour_masks_in_registered_roi

def metric(heat,mask):
 h=np.maximum(np.asarray(heat,float),0);total=h.sum();flat=h.ravel(); out={'energy':None if total<=0 else float(h[mask].sum()/total),'pointing':bool(mask[np.unravel_index(int(h.argmax()),h.shape)])}
 for f in (.1,.2):
  n=max(1,int(np.ceil(flat.size*f)));chosen=h>=np.partition(flat,-n)[-n];out[f'top{int(f*100)}_overlap']=float((chosen&mask).sum()/max((chosen|mask).sum(),1))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sel=json.loads(a.selection.read_text());raw=next((a.data_root/'eyes/papila/raw').glob('PapilaDB-PAPILA-*'));cont=raw/'ExpertsSegmentations/Contours'; rows=[]
 for group,key in ((sel['cases'],'binary'),(sel['suspect_cases'],'suspect')):
  for sid,item in group.items():
   sample=item.get('sample_id');
   if not sample:continue
   c=a.cases/sample
   if not c.is_dir():continue
   masks=contour_masks_in_registered_roi(raw/'FundusImages'/f'{sample}.jpg',cont/f'{sample}_disc_exp1.txt',{n:cont/f'{sample}_{n}_exp1.txt' for n in ('disc','cup')}); masks['rim']=masks['disc']&~masks['cup']
   for method,file in [('lime_positive_support','lime_positive_map.npy'),('grad_cam','grad_cam_raw.npy')]:
    h=np.load(c/file)
    for structure,mask in masks.items():
     d=metric(h,mask);rows.append({'selection_id':sid,'cohort':key,'physical_case_id':sample,'method':method,'expert_mask_source':'expert1','structure':structure,**d,'semantics':'spatial correspondence diagnostic; not Gamma or causal proof'})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(a.output)
if __name__=='__main__':main()

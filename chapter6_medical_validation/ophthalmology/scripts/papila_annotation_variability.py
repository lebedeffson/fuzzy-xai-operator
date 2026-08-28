from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
def fill(path):
 p=np.loadtxt(path,dtype=np.float32);m=np.zeros((1934,2576),np.uint8);cv2.fillPoly(m,[np.rint(p).astype(np.int32)],1);return m.astype(bool)
def dice(a,b):return float(2*(a&b).sum()/max(a.sum()+b.sum(),1))
def cdr(cup,disc):return float(np.sqrt(cup.sum()/max(disc.sum(),1)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.selection.read_text());root=next((a.data_root/'eyes/papila/raw').glob('PapilaDB-PAPILA-*'))/'ExpertsSegmentations/Contours';out=[]
 for group in (s['cases'],s['suspect_cases']):
  for alias,item in group.items():
   sid=item.get('sample_id');
   if not sid:continue
   paths={f'{z}{e}':root/f'{sid}_{z}_exp{e}.txt' for z in ('disc','cup') for e in (1,2)}
   if not all(x.is_file() for x in paths.values()):continue
   d1,d2,c1,c2=(fill(paths[k]) for k in ('disc1','disc2','cup1','cup2'));r1,r2=cdr(c1,d1),cdr(c2,d2);out.append({'selection_id':alias,'physical_case_id':sid,'disc_dice':dice(d1,d2),'cup_dice':dice(c1,c2),'cdr_expert1':r1,'cdr_expert2':r2,'cdr_absolute_difference':abs(r1-r2),'cdr_formula':'sqrt(area(cup)/area(disc))','semantics':'expert annotation variability diagnostic; not Gamma or expert error'})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(a.output)
if __name__=='__main__':main()

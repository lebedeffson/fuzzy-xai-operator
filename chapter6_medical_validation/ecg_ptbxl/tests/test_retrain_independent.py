"""Technical unit tests; toy tensors are not ECG experimental evidence."""
import numpy as np
import torch
from chapter6_medical_validation.ecg_ptbxl.scripts.evaluate_retrain_independent import independent_ig, independent_occlusion, rank_mask, assess_review

class Linear(torch.nn.Module):
    def forward(self,x):
        z=x.sum((1,2))/12000
        return torch.stack((-z,z),dim=1)

def test_linear_ig_and_occlusion():
    x=torch.ones(1,12,1000)
    ig, completeness=independent_ig(Linear(),x,1)
    assert ig.shape==(12,1000)
    assert abs(ig.sum()-1)<1e-5
    assert completeness["completeness_residual"]<1e-5
    occ=independent_occlusion(Linear(),x,1,1.)
    assert occ.shape==(12,20)
    assert np.all(occ>0)
    assert np.allclose(occ,occ[0,0],atol=1e-6)

def test_budget_and_tie():
    assert rank_mask([.4,.4,.9],[2,1,3],2).tolist()==[False,True,True]
    rows=[{"ecg_id":i,"prediction":i%2,"truth":int(i%3==0),"p_abnormal":.9 if i%2 else .1,"technical_quality_score":.9,"rho":i/10,"patient_hash":str(i//2),"action":"accept"} for i in range(10)]
    r=assess_review(rows,bootstrap=50)
    assert r["review_n_each"]==2
    assert all(x["review_n"]==2 and x["accept_n"]==8 and x["flagged_errors"]+x["remaining_errors"]==r["total_errors"] for x in r["metrics"].values())

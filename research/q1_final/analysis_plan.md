# Q1 final analysis plan

The final cycle is independent of the frozen Q1 remediation evidence at commit
`41c32af25242164144fd907e4850fa9d4f426bd1`. Controlled, real, human and domain evidence are reported separately.

All model and policy selection uses train/validation partitions. Test is used once for the frozen primary analysis.
Native multiclass tasks retain their original labels. Explanation methods share a preregistered stratified object set.
Paired confidence intervals use deterministic bootstrap seeds; secondary endpoints use Holm correction.

H3 is not a universal-superiority hypothesis. Full-population and hard-case claims are separate. H5 predictive gain
is not required for release and remains `not_supported` when M1 does not improve M0. H6 may end as
`not_supported`; in that case rule ablation remains a local diagnostic only.

External study software never generates participant, reviewer, consent or signature records. Negative external
results permit release only after the associated benefit claim is removed. Missing external records keep the gate
`open` and stable release disabled.

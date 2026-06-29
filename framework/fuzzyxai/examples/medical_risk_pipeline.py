from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzyxai import ExplainPlan, HesitantFS, IntervalFS, MultiLevelFS, NeutrosophicFS
from fuzzyxai import Rule, SystemOperator, Trace, build_profile, compose, interpretability_index, interpretability_loss, select_minimal_sufficient, Candidate


def main():
    plan = ExplainPlan().with_reduction_weight(0.10)
    metadata = {'has_intervals': True, 'num_experts': 2, 'source_conflict': True, 'requires_audit': True, 'multi_level': True}
    profile = build_profile(metadata)
    candidates = [
        Candidate('F0', {'u_num','u_ling'}, 0.10, 0.03, 0.42),
        Candidate('FI', {'u_num','u_ling','u_int'}, 0.22, 0.05, 0.30),
        Candidate('FH', {'u_num','u_ling','u_exp'}, 0.30, 0.81, 0.25),
        Candidate('FNsrc', {'u_num','u_ling','u_if','u_conf','u_trace'}, 0.42, 0.09, 0.18),
        Candidate('FML-audit', {'u_num','u_ling','u_int','u_exp','u_conf','u_trace','u_multi'}, 0.68, 1.00, 0.04),
    ]
    selected = select_minimal_sufficient(profile, candidates, mode='audit')
    print('Profile:', sorted(profile))
    print('Selected:', selected)

    interval = IntervalFS(lambda x: 0.68, lambda x: 0.76)
    hesitant = HesitantFS(lambda x: [0.61, 0.78])
    neutro = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64, source_t='model', source_f='expert')
    rep = MultiLevelFS([interval, hesitant, neutro], gamma={('interval','hesitant','same_case'), ('neutro','trace','source_link')}, weights=[0.25,0.25,0.50])
    _, delta = rep.reduce_to_f0()
    print('Reduction loss Delta:', round(delta, 4))

    op = SystemOperator(plan)
    e1 = op.explain_scalar_risk(0.72, [Rule('r_high_check', {'risk':'high'}, 'additional_check'), Rule('r_medium_watch', {'risk':'medium'}, 'watch')], Trace('case-001','risk-v1','2026-05-22T10:00:00', source='demo', checksum='abc'))
    e1.representation = rep
    e1.reduction_loss = delta
    e2 = op.explain_scalar_risk(0.74, [Rule('r_decision_high', {'risk':'high'}, 'send_to_check')], Trace('case-001-decision','decision-v1','2026-05-22T10:00:01', source='demo', checksum='def'), model_uncertainty=0.10, trace_uncertainty=0.01)
    composed = compose(e1, e2, plan.beta)
    print('Composed type:', type(composed).__name__)
    if hasattr(composed, 'uncertainty'):
        loss = interpretability_loss(0.34, 0.40, 0.18, 0.05, composed.uncertainty, plan.lambda_, composed.reduction_loss, lambda_delta=0.10)
        print('Gamma:', round(composed.metadata.get('gamma', 0.0), 4))
        print('L_ext:', round(loss, 4))
        print('I:', round(interpretability_index(loss), 4))

if __name__ == '__main__':
    main()

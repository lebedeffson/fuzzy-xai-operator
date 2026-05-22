from fuzzyxai import build_profile, select_minimal_sufficient, Candidate

def test_selector_chooses_audit_multilevel():
    profile = build_profile({'has_intervals': True, 'num_experts': 2, 'source_conflict': True, 'requires_audit': True})
    candidates = [Candidate('F0', {'u_num','u_ling'}, 0.1, 0.03, 0.42), Candidate('FML-audit', {'u_num','u_ling','u_int','u_exp','u_conf','u_trace','u_multi'}, 0.68, 1.0, 0.04)]
    selected = select_minimal_sufficient(profile, candidates, 'audit')
    assert selected.name == 'FML-audit'

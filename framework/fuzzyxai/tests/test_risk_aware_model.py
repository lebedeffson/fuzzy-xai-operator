from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fuzzyxai import ExplainPlan
from fuzzyxai.risk import RiskAction, RiskAwareModel, RiskPolicy


def _fit_observer():
    data = load_breast_cancer(as_frame=True)
    x = data.data
    y = (data.target == 0).astype(int)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42).fit(x_train, y_train)
    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    observer = RiskAwareModel(model, plan=plan, policy=RiskPolicy(theta_mid=0.35, theta_high=0.65), positive_class=1)
    return observer, x_test


def test_risk_aware_model_returns_required_fields():
    observer, x_test = _fit_observer()
    rows = observer.predict_with_risk(x_test.iloc[:3])
    assert len(rows) == 3
    for row in rows:
        assert 0 <= row['predicted_risk'] <= 1
        assert 0 <= row['uncertainty'] <= 1
        assert row['action'] in {action.value for action in RiskAction}
        assert 'interpretability_index' in row


def test_risk_aware_model_forced_diagnostic_blocks():
    observer, x_test = _fit_observer()
    row = observer.predict_with_risk(x_test.iloc[:1], metadata={'force_diagnostic': True})[0]
    assert row['action'] == RiskAction.BLOCK.value
    assert row['diagnostics'] == ['D_ij']

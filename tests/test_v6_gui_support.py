from pathlib import Path

from fuzzyxai.reporting.session_report import SessionReport
from fuzzyxai.text.explanation_generator import generate_explanation_with_optional_llm


def test_session_report_writes_csv(tmp_path):
    log_path = tmp_path / 'session.csv'
    report = SessionReport('demo', log_path=log_path)
    report.add_step('step', {'value': 1})
    assert log_path.exists()
    text = log_path.read_text(encoding='utf-8')
    assert 'timestamp,name,payload_json' in text
    assert 'step' in text


def test_optional_llm_falls_back_without_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    text, trace = generate_explanation_with_optional_llm({'risk': 0.72, 'selected_class': 'FML-audit'}, use_llm=True)
    assert '0.72' in text
    assert trace['mode'] == 'template'

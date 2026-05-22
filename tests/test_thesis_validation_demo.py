from proofs.validate_thesis_examples import build_validation_report
from examples.thesis_demo import run_demo


def test_thesis_validation_passes():
    report = build_validation_report()
    assert report["status"] == "PASS"
    assert report["failed_checks"] == 0
    assert report["total_checks"] >= 20


def test_thesis_demo_route_passes():
    report = run_demo()
    assert report["status"] == "PASS"
    assert "P_sit" in report["route"]
    assert len(report["timeline"]) >= 8

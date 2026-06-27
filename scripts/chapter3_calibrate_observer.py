#!/usr/bin/env python3
from chapter3_evidence_common import calibrate_observer

if __name__ == "__main__":
    data = calibrate_observer()
    print("Saved reports/chapter3/observer_calibration_report.json")
    print("Best objective:", data["best_config_metrics_validation"]["objective_J"])

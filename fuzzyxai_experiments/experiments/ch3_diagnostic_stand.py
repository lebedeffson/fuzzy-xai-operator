from __future__ import annotations
import json
from fuzzyxai_experiments.src.report_builders import build_ch3_critical_ruptures
from fuzzyxai_experiments.src.utils import timed_report

if __name__ == '__main__':
    print(json.dumps(timed_report('ch3_diagnostic_stand', build_ch3_critical_ruptures), ensure_ascii=False, indent=2))

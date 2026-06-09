from __future__ import annotations
import json
from fuzzyxai_experiments.src.report_builders import build_ch2_critical_ruptures
from fuzzyxai_experiments.src.utils import timed_report

if __name__ == '__main__':
    print(json.dumps(timed_report('ch2_critical_ruptures', build_ch2_critical_ruptures), ensure_ascii=False, indent=2))

from __future__ import annotations
import json
from fuzzyxai_experiments.src.report_builders import build_ch5_hybrid
from fuzzyxai_experiments.src.utils import timed_report

if __name__ == '__main__':
    print(json.dumps(timed_report('ch5_hybrid', build_ch5_hybrid), ensure_ascii=False, indent=2))

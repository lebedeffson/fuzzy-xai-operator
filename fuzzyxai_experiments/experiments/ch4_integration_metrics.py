from __future__ import annotations
import json
from fuzzyxai_experiments.src.report_builders import build_ch4_integration
from fuzzyxai_experiments.src.utils import timed_report

if __name__ == '__main__':
    print(json.dumps(timed_report('ch4_integration', build_ch4_integration), ensure_ascii=False, indent=2))

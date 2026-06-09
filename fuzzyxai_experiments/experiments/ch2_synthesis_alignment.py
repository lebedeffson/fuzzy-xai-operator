from __future__ import annotations
import json
from fuzzyxai_experiments.src.report_builders import build_ch2_synthesis
from fuzzyxai_experiments.src.utils import timed_report

if __name__ == '__main__':
    print(json.dumps(timed_report('ch2_synthesis', build_ch2_synthesis), ensure_ascii=False, indent=2))

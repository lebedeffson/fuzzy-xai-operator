#!/usr/bin/env python3
from chapter3_evidence_common import build_real_conflicts

if __name__ == "__main__":
    df = build_real_conflicts()
    print(f"Saved reports/chapter3/real_conflict_summary.csv ({len(df)} rows)")

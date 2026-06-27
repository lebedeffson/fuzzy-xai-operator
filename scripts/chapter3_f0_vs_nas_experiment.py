#!/usr/bin/env python3
from chapter3_evidence_common import f0_vs_nas

if __name__ == "__main__":
    df = f0_vs_nas()
    print("Saved reports/chapter3/f0_vs_nas_action_diff.csv")
    print(df.to_string(index=False))

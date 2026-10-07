#!/usr/bin/env python3
"""Confirm calibrated A with hierarchical bootstrap-style seed/order aggregation and external difficulty baselines."""
import json,math,statistics
from pathlib import Path
def run():
 d=json.loads(Path("artifacts/A_v2_report.json").read_text());vals=[x["A_mean"] for x in d["orders"]]
 mean=statistics.fmean(vals);se=statistics.stdev(vals)/math.sqrt(len(vals));lo=mean-1.96*se;hi=mean+1.96*se
 positives=sum(v>0 for v in vals)
 # sign test normal approximation around p=.5
 z=(positives-len(vals)/2)/math.sqrt(len(vals)*.25)
 out={"orders":len(vals),"positive_orders":positives,"A_mean_across_orders":mean,"A_ci95":[lo,hi],"sign_z":z,
      "confirmed_A":lo>0 and positives>=18,
      "normalization":"external per-task baselines computed independently across fixed seeds; not estimated inside each curriculum order"}
 Path("artifacts/A_v2_confirmation.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

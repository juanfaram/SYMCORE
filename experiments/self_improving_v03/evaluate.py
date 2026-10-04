#!/usr/bin/env python3
"""Hard gates: compare evolution against causal baselines."""
import csv,json,subprocess,sys
from pathlib import Path
def baselines(path):
    last=None; by_hour={}; ae_last=[];ae_hour=[]
    with path.open() as f:
      for r in csv.DictReader(f):
        y=float(r["cnt"]);h=r["hr"]
        if last is not None:ae_last.append(abs(last-y))
        if h in by_hour:ae_hour.append(abs(by_hour[h]-y))
        by_hour[h]=y;last=y
    return {"persistence_mae":sum(ae_last[-512:])/len(ae_last[-512:]),
            "same_hour_mae":sum(ae_hour[-512:])/len(ae_hour[-512:])}
if __name__=="__main__":
    from main import ensure,run
    p=Path("data/hour.csv");ensure(p);report=run(p,Path("artifacts"),seed=17)
    b=baselines(p);out={"champion_mae":report["champion_mae"],**{k:round(v,3) for k,v in b.items()}}
    out["beats_persistence"]=out["champion_mae"]<b["persistence_mae"]
    out["beats_same_hour"]=out["champion_mae"]<b["same_hour_mae"]
    Path("artifacts/baseline_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

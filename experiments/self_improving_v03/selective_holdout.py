#!/usr/bin/env python3
"""Pre-registered temporal holdout test for selective intervention value concentration."""
import csv,json,math,random,statistics
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from intervention_controller import InterventionController

def block_bootstrap_R(delta,g,start,block=256,reps=2000,seed=20261005):
    xs=list(range(start,len(delta),block));rng=random.Random(seed);vals=[]
    total=sum(delta[start:]);rate=sum(g[start:])/max(1,len(g)-start)
    if not xs or rate==0 or total<=0:return []
    for _ in range(reps):
        picked=[rng.choice(xs) for __ in xs];num=0.;den=0.
        for b in picked:
            e=min(len(delta),b+block);num+=sum(delta[i]*g[i] for i in range(b,e));den+=sum(delta[b:e])
        if den>0:vals.append((num/max(1e-12,rate*den)))
    return sorted(vals)

def run():
    path=Path("data/hour.csv");ensure(path)
    rows=list(csv.DictReader(path.open()));split=int(len(rows)*.60)
    online=Host(("hr","workingday"));always=SymcoreHost();ctl=InterventionController()
    delta=[];g=[]
    # First 60% is calibration/warm-up only. Controller constants remain frozen.
    for i,r in enumerate(rows):
        y=float(r["cnt"]);active=ctl.decide();oe=online.step(r,y);se=always.step(r,y)
        delta.append(oe-se);g.append(int(active));ctl.observe(oe)
    d=delta[split:];gg=g[split:];rate=sum(gg)/len(gg);total=sum(d);captured=sum(x*y for x,y in zip(d,gg))
    R=captured/max(1e-12,rate*total) if rate>0 and total>0 else None
    bs=block_bootstrap_R(delta,g,split)
    ci=[bs[int(.025*len(bs))],bs[int(.975*len(bs))-1]] if bs else None
    out={"schema":"symcore.selective_holdout.v1","rows":len(rows),"calibration_rows":split,
         "holdout_rows":len(d),"controller_frozen":True,"active_fraction_holdout":round(rate,6),
         "total_motor_value":round(total,6),"captured_value":round(captured,6),
         "R":round(R,6) if R is not None else None,
         "R_block_bootstrap_ci95":[round(x,6) for x in ci] if ci else None,
         "endpoint":"R > 1 with block-bootstrap CI95 excluding 1"}
    Path("artifacts").mkdir(exist_ok=True);Path("artifacts/selective_holdout_report.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

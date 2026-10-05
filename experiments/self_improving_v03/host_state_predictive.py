#!/usr/bin/env python3
"""One-round causal validation of HostState: can state at t predict FUTURE motor value t+1..t+H?"""
import csv,json,math,statistics
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from host_state import HostState
def auc(scores,labels):
 pos=[s for s,y in zip(scores,labels) if y];neg=[s for s,y in zip(scores,labels) if not y]
 if not pos or not neg:return .5
 return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def run(H=12):
 p=Path("data/hour.csv");ensure(p);rows=list(csv.DictReader(p.open()));o=Host(("hr","workingday"));m=SymcoreHost();hs=HostState();states=[];delta=[]
 for r in rows:
  states.append(hs.vector());y=float(r["cnt"]);oe=o.step(r,y);me=m.step(r,y);delta.append(oe-me);hs.observe(oe,me,[oe,me])
 X=[];Y=[]
 for t in range(64,len(rows)-H):
  X.append(states[t]);Y.append(statistics.fmean(delta[t+1:t+H+1]))
 split=int(len(X)*.60);keys=list(X[0]);mu={k:statistics.fmean(x[k] for x in X[:split]) for k in keys};sd={k:statistics.pstdev([x[k] for x in X[:split]]) or 1 for k in keys}
 # direction learned on calibration only; equal-weight standardized causal score
 corr={}
 for k in keys:
  z=[(x[k]-mu[k])/sd[k] for x in X[:split]];ym=statistics.fmean(Y[:split]);corr[k]=sum(a*(b-ym) for a,b in zip(z,Y[:split]))
 score=lambda x:sum((1 if corr[k]>=0 else -1)*(x[k]-mu[k])/sd[k] for k in keys)
 threshold=statistics.median(Y[:split]);scores=[score(x) for x in X[split:]];labels=[y>threshold for y in Y[split:]]
 A=auc(scores,labels)
 out={"horizon":H,"split":.60,"features":keys,"holdout_auc":A,"predictive":A>=.60,"endpoint":"holdout AUC >= 0.60; one-round decision"}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/host_state_predictive.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

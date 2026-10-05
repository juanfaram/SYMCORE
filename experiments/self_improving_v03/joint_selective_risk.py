#!/usr/bin/env python3
import csv,json
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from intervention_controller import InterventionController
from risk_contract import risk_report,passes
from selective_holdout import block_bootstrap_R
def run(rmax=.05,delta_budget=.05):
 p=Path("data/hour.csv");ensure(p);rows=list(csv.DictReader(p.open()));split=int(len(rows)*.60)
 online=Host(("hr","workingday"));motor=SymcoreHost();ctl=InterventionController();d=[];g=[];control=[];selective=[]
 for r in rows:
  y=float(r["cnt"]);active=ctl.decide();oe=online.step(r,y);se=motor.step(r,y);ctl.observe(oe);d.append(oe-se);g.append(int(active));control.append(oe);selective.append(se if active else oe)
 hd=d[split:];hg=g[split:];hc=control[split:];hs=selective[split:];rate=sum(hg)/len(hg);total=sum(hd);captured=sum(x*y for x,y in zip(hd,hg))
 R=captured/max(1e-12,rate*total) if rate and total>0 else None;bs=block_bootstrap_R(d,g,split);ci=[bs[int(.025*len(bs))],bs[int(.975*len(bs))-1]] if bs else None
 risk=risk_report(hc,hs,rmax);gain=sum(a-b for a,b in zip(hc,hs))/len(hc)
 out={"split":.60,"controller_frozen":True,"active_fraction":rate,"mean_gain":gain,"R":R,"R_ci95":ci,"risk":risk,
      "joint_pass":gain>0 and passes(risk,delta_budget) and ci is not None and ci[0]>1,
      "contract":"mean_gain>0 AND risk_upper95<delta AND R_CI95_lower>1"}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/joint_selective_risk.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

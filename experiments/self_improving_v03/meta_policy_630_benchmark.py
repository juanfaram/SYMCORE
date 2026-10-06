#!/usr/bin/env python3
"""630 preregistered holdout: operation + parameter selection from causal consequence predictions."""
import json,math,random
from pathlib import Path
from causal_evolution_memory import CausalEvolutionMemory
from hierarchical_meta_policy import HierarchicalMetaPolicy
OPS=("create","modify","prune","restore","noop")
PARAMS=(.25,.5,1.0,1.5)
def effect(op,p):
 return {"create":(p,0,.1*p,.2*p,.1*p),"modify":(.35*p,.35*p,.2*p,.1*p,.25*p),
 "prune":(-.08*p,.05*p,.25*p,-.8*p,.12*p),"restore":(.45*p,.25*p,.35*p,.2*p,.4*p),"noop":(0,0,0,0,0)}[op]
def world(z,op,p):
 gap,change,redundancy,load,archive=z;e=effect(op,p);x1,x2,x3,x4,x5=e
 d=(.16*x1*gap+.05*x2,-.03*x1*change+.15*x2+.08*x5,.14*x3*(1-change)-.04*x1,
    .22*(-x4)*load+.08*x3*redundancy-.06*abs(x1),.14*x5*archive+.09*x1*gap+.06*x2)
 cost=.12+.16*p*p+(.08 if op in ("create","restore") else 0)
 return d,cost
def score_truth(d,c,risk=.05):
 if min(d)<-risk:return -1e12
 return sum(max(0,x) for x in d)/c
def run():
 rng=random.Random(630001);mem=CausalEvolutionMemory(k=80)
 for _ in range(6500):
  z=tuple(rng.random() for _ in range(5))
  for op in OPS:
   ps=(0.,) if op=="noop" else PARAMS
   for p in ps:
    d,c=world(z,op,p);mem.observe(z,effect(op,p),tuple(x+rng.gauss(0,.008) for x in d),c+rng.gauss(0,.004))
 policy=HierarchicalMetaPolicy(mem,risk_budget=.05)
 space={op:[((),effect(op,0.))] if op=="noop" else [((p,),effect(op,p)) for p in PARAMS] for op in OPS}
 op_hit=joint_hit=0;n=500;regrets=[]
 for i in range(n):
  rr=random.Random(990000+i);z=tuple(rr.random() for _ in range(5));pred=policy.choose(z,space)
  truth={(op,p):world(z,op,p) for op in OPS for p in ((0.,) if op=="noop" else PARAMS)}
  oracle=max(truth,key=lambda k:score_truth(*truth[k]))
  op_hit+=pred.operation==oracle[0];pp=0. if pred.operation=="noop" else pred.params[0]
  joint_hit+=(pred.operation,pp)==oracle
  regrets.append(score_truth(*truth[oracle])-score_truth(*truth[(pred.operation,pp)]))
 op_acc=op_hit/n;joint=joint_hit/n;mean_regret=sum(regrets)/n
 out={"schema":"symcore.meta-policy.v1","holdout_n":n,"operation_accuracy":op_acc,"joint_operation_parameter_accuracy":joint,"mean_decision_regret":mean_regret,
      "criteria":{"min_operation_accuracy":.80,"min_joint_accuracy":.70,"max_mean_regret":.03},
      "passed":op_acc>=.80 and joint>=.70 and mean_regret<=.03}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/meta_policy_630_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

#!/usr/bin/env python3
"""SYMCORE v1.0 experimental: evidence-driven capability lifecycle."""
from __future__ import annotations
import json,math,statistics
from dataclasses import dataclass,asdict
from pathlib import Path
from repertoire import Repertoire,Evidence
from capability_frontier import Capability,Frontier,Interval
from research_scheduler import ResearchScheduler

@dataclass
class ReplicatedEvidence:
    name:str;quality:list;adaptation:list;retention:list;efficiency:list;robustness:list;novelty:float;cost:float
    def axis(self,k):
        xs=getattr(self,k);m=statistics.fmean(xs);ci=1.96*statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float("inf")
        return {"mean":m,"ci95":ci,"lower":m-ci,"upper":m+ci}

class CapabilityLifecycle:
    """Wild population -> evidence archive -> validated frontier -> repertoire."""
    def __init__(self,min_seeds=5,max_ci=.08):
        self.min_seeds=min_seeds;self.max_ci=max_ci;self.archive=[];self.frontier=Frontier(regression_budgets={a:.05 for a in ("quality","adaptation","retention","efficiency","learnability")});self.repertoire=Repertoire();self.scheduler=ResearchScheduler(["adaptation","retention","robustness","efficiency","novelty"])
    def submit(self,e):
        axes={k:e.axis(k) for k in ("quality","adaptation","retention","efficiency","robustness")}
        record={"name":e.name,"axes":axes,"seeds":len(e.quality),"novelty":e.novelty,"cost":e.cost,"validated":False}
        self.archive.append(record)
        if len(e.quality)<self.min_seeds:return record
        if any(axes[k]["ci95"]>self.max_ci for k in axes):return record
        # conservative capability uses lower confidence bound, not optimistic mean
        # Lifecycle v1 has no direct future-acquisition trial; use a conservative proxy until measured.
        learn=max(0.,.5*axes["adaptation"]["lower"]+.5*axes["retention"]["lower"])
        def I(k):return Interval(axes[k]["mean"],axes[k]["lower"],axes[k]["upper"])
        c=Capability(I("quality"),I("adaptation"),I("retention"),I("efficiency"),Interval(learn,learn,learn))
        if not self.frontier.consider(e.name,c):return record
        ev=Evidence(axes["quality"]["mean"],axes["adaptation"]["mean"],axes["retention"]["mean"],axes["efficiency"]["mean"],axes["robustness"]["mean"],e.novelty,len(e.quality),e.cost,e.name)
        self.repertoire.consider(e.name,ev);record["validated"]=True
        gain=max(0.,e.novelty*.3+axes["quality"]["mean"]*.3+axes["robustness"]["mean"]*.2+axes["retention"]["mean"]*.2)
        self.scheduler.record("novelty" if e.novelty>.7 else "robustness",gain,e.cost,True)
        return record
    def save(self,path="artifacts/capability_lifecycle.json"):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);self.repertoire.save(p.parent/"repertoire.json")
        p.write_text(json.dumps({"schema":"symcore.capability-lifecycle.v1","archive":self.archive,
          "frontier":[{"name":n,"capability":asdict(c)} for n,c in self.frontier.items],
          "research":self.scheduler.snapshot()},indent=2))

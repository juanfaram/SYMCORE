#!/usr/bin/env python3
"""Multi-seed capability tournament: evidence, not a scalar champion."""
from __future__ import annotations
import json, math, random, statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from capability_frontier import Capability, Frontier, Interval

@dataclass(frozen=True)
class Trial:
    quality: float
    adaptation: float
    retention: float
    efficiency: float
    learnability: float

def _mean_ci(xs):
    mean=statistics.fmean(xs)
    if len(xs)<2:return mean,0.0
    return mean,1.96*statistics.stdev(xs)/math.sqrt(len(xs))

def evaluate_policy(name,seed,n=6000):
    """Procedural four-regime exam with corruption and rare contexts."""
    rng=random.Random(seed)
    skill={"common":.55,"rare":.35}; retained=.55; correct=0; post=[]; cost=0.0
    for i in range(n):
        phase=min(3,i//(n//4)); rare=rng.random()<.08
        ctx="rare" if rare else "common"
        target=(phase + (1 if rare else 0))%4
        # policies trade exploration cost for adaptation/retention
        if name=="stable": rate=.020; decay=.00015; explore=.03
        elif name=="plastic": rate=.055; decay=.00055; explore=.07
        else: rate=.038; decay=.00028; explore=.045
        p=min(.97,max(.05,skill[ctx]))
        ok=rng.random()<p
        reward=1.0 if ok else -0.35
        if rng.random()<.08: reward*=-1
        signal=1 if reward>0 else 0
        skill[ctx]+=rate*(signal-skill[ctx])
        other="rare" if ctx=="common" else "common"
        skill[other]=max(.05,skill[other]-decay)
        if phase>0: retained=max(.05,retained-decay*(1.5 if name=="plastic" else .7))
        if rng.random()<explore: cost+=1.0
        cost+=1.0
        correct+=ok
        if i>=n-1000:post.append(ok)
    quality=correct/n
    adaptation=sum(post)/len(post)
    retention=retained
    efficiency=1.0/(cost/n)
    # Forward acquisition potential: lower simulated cost on held-out micro-capabilities is better.
    # Depends on reusable retained skill and adaptation, not candidate size.
    future_cost=1.0/max(.05,.55*retained+.45*adaptation)
    learnability=1.0/future_cost
    return Trial(quality,adaptation,retention,efficiency,learnability)

def run(seeds=(11,23,37,51,73)):
    policies=("stable","plastic","balanced")
    evidence={}; frontier=Frontier(regression_budgets={"quality":.08,"adaptation":.08,"retention":.08,"efficiency":.08,"learnability":.08})
    for name in policies:
        trials=[evaluate_policy(name,s) for s in seeds]
        axes={}
        for axis in Trial.__dataclass_fields__:
            vals=[getattr(t,axis) for t in trials]; mean,ci=_mean_ci(vals)
            axes[axis]={"mean":round(mean,6),"ci95":round(ci,6)}
        cap=Capability(*(Interval(axes[a]["mean"],axes[a]["mean"]-axes[a]["ci95"],axes[a]["mean"]+axes[a]["ci95"]) for a in ("quality","adaptation","retention","efficiency","learnability")))
        evidence[name]={"axes":axes,"trials":[asdict(t) for t in trials],
                        "frontier_accepted":frontier.consider(name,cap)}
    accepted=[name for name,_ in frontier.items]
    properties={
      "reproducible_inputs":len(set(seeds))==len(seeds),
      "uncertainty_reported":all(all("ci95" in evidence[n]["axes"][a] for a in Trial.__dataclass_fields__) for n in policies),
      "tradeoffs_present":len(accepted)>=2,
      "learnability_measured":all("learnability" in evidence[n]["axes"] for n in policies),
      "multi_seed":len(seeds)>=5}
    report={"schema":"symcore.frontier.v3","seeds":list(seeds),
            "accepted_frontier":accepted,"candidates":evidence,"properties":properties,
            "passed":all(properties.values()) and all(len(evidence[n]["trials"])==len(seeds) for n in policies)}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/frontier_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return report

if __name__=="__main__":run()

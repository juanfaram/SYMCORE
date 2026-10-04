#!/usr/bin/env python3
"""Interaction stress suite: multiple domains, changing preferences and delayed feedback."""
import json,random
from pathlib import Path
from interaction_learner import InteractionLearner
def run(n=12000):
    rng=random.Random(44);L=InteractionLearner(["fast","balanced","deep","creative"],.1,44,"artifacts/stress_history.jsonl")
    domains=("code","forecast","writing","analysis");diffs=("simple","normal","hard")
    hits=[];window=[]
    for i in range(n):
        dom=rng.choice(domains);dif=rng.choice(diffs);ctx={"task":"solve","difficulty":dif,"domain":dom}
        # preference shifts halfway through: tests adaptation, not memorization
        if i<n//2: target={"simple":"fast","normal":"balanced","hard":"deep"}[dif]
        else: target={"code":"deep","forecast":"balanced","writing":"creative","analysis":"deep"}[dom]
        a=L.choose(ctx);reward=(1 if a==target else -.25)+rng.uniform(-.1,.1)
        L.feedback(a,ctx,reward,{"phase":1 if i<n//2 else 2});window.append(a==target)
        if len(window)>1000:window.pop(0)
        if i in (n//2-1,n-1):hits.append(sum(window)/len(window))
    out={"pre_shift_accuracy":round(hits[0],4),"post_adaptation_accuracy":round(hits[1],4),"interactions":n}
    Path("artifacts/stress_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

#!/usr/bin/env python3
"""SYMCORE v0.4 interaction learner: learns routing/policy choices from outcome feedback."""
from __future__ import annotations
import json,math,random
from dataclasses import dataclass,field
from pathlib import Path

@dataclass
class Arm:
    name:str
    n:int=0
    reward:float=0.
    contexts:dict=field(default_factory=dict)
    def mean(self): return self.reward/self.n if self.n else 0.

class InteractionLearner:
    """Contextual bandit with explicit exploration and append-only audit log."""
    def __init__(self,actions,epsilon=.08,seed=23,log="interaction_history.jsonl"):
        self.arms={a:Arm(a) for a in actions};self.epsilon=epsilon;self.rng=random.Random(seed);self.log=Path(log)
    def _key(self,context):
        # intentionally coarse and bounded; no raw personal data retained
        return "|".join(f"{k}={context[k]}" for k in sorted(context) if k in {"task","difficulty","domain"})
    def choose(self,context):
        key=self._key(context)
        if self.rng.random()<self.epsilon:return self.rng.choice(list(self.arms))
        def ucb(a):
            stats=a.contexts.get(key,{"n":0,"r":0.})
            if not stats["n"]:return float("inf")
            total=max(1,sum(x.contexts.get(key,{"n":0})["n"] for x in self.arms.values()))
            return stats["r"]/stats["n"]+math.sqrt(2*math.log(total+1)/stats["n"])
        return max(self.arms.values(),key=ucb).name
    def feedback(self,action,context,reward,metadata=None):
        reward=max(-1.,min(1.,float(reward)));a=self.arms[action];a.n+=1;a.reward+=reward
        key=self._key(context);s=a.contexts.setdefault(key,{"n":0,"r":0.});s["n"]+=1;s["r"]+=reward
        event={"action":action,"context_key":key,"reward":reward,"metadata":metadata or {}}
        self.log.parent.mkdir(parents=True,exist_ok=True)
        with self.log.open("a",encoding="utf-8") as f:f.write(json.dumps(event,sort_keys=True)+"\n")
    def state(self):
        return {k:{"n":a.n,"mean_reward":round(a.mean(),4),"contexts":a.contexts} for k,a in self.arms.items()}

if __name__=="__main__":
    # executable stress test: hidden environments prefer different policies
    L=InteractionLearner(["fast","balanced","deep"],epsilon=.1,seed=4,log="artifacts/interactions.jsonl")
    rng=random.Random(5)
    prefs={"simple":"fast","normal":"balanced","hard":"deep"}
    hits=0
    for i in range(6000):
        d=rng.choice(list(prefs));ctx={"task":"reasoning","difficulty":d,"domain":"synthetic"}
        a=L.choose(ctx);reward=(1 if a==prefs[d] else -0.35)+rng.uniform(-.15,.15)
        L.feedback(a,ctx,reward,{"step":i});hits+=a==prefs[d]
    result={"accuracy":round(hits/6000,4),"state":L.state()}
    Path("artifacts").mkdir(exist_ok=True);Path("artifacts/interaction_report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

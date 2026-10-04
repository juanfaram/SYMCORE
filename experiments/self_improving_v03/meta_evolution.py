#!/usr/bin/env python3
"""SYMCORE meta-evolution: experience-guided mutation operator selection."""
from __future__ import annotations
import json,math,random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from main import Genome

OPS=("lr_up","lr_down","toggle_features","momentum","regularize","clip")

class EvolutionMemory:
    def __init__(self,path="artifacts/meta_memory.json",seed=31):
        self.path=Path(path)
        self.rng=random.Random(seed)
        self.stats=defaultdict(lambda:{"n":0,"reward":0.0})
        if self.path.exists():
            for k,v in json.loads(self.path.read_text()).items():
                self.stats[k]=v

    def choose(self,context):
        total=sum(self.stats[f"{context}:{o}"]["n"] for o in OPS)+1
        unseen=[o for o in OPS if self.stats[f"{context}:{o}"]["n"]==0]
        if unseen:
            return self.rng.choice(unseen)
        def ucb(op):
            s=self.stats[f"{context}:{op}"]
            mean=s["reward"]/s["n"]
            explore=math.sqrt(2*math.log(total)/s["n"])
            return mean+explore
        return max(OPS,key=ucb)

    def record(self,context,op,reward):
        s=self.stats[f"{context}:{op}"]
        s["n"]+=1
        s["reward"]+=max(-1.0,min(1.0,float(reward)))

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(dict(self.stats),indent=2))

def apply(g,op,rng):
    d=asdict(g)
    if op=="lr_up": d["lr"]=min(1.,g.lr*rng.uniform(1.1,1.8))
    elif op=="lr_down": d["lr"]=max(.002,g.lr/rng.uniform(1.1,1.8))
    elif op=="toggle_features": d["rich"]=not g.rich
    elif op=="momentum": d["momentum"]=max(0.,min(.95,g.momentum+rng.uniform(-.25,.25)))
    elif op=="regularize": d["l2"]=max(0.,min(.05,g.l2+rng.uniform(-.005,.005)))
    elif op=="clip": d["clip"]=max(50.,min(1000.,g.clip*rng.uniform(.7,1.3)))
    return Genome(**d)

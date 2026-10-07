#!/usr/bin/env python3
"""SYMCORE v0.9: quality-diversity archive and capability repertoire."""
from dataclasses import dataclass,asdict
import json,math
from pathlib import Path
@dataclass
class Evidence:
    quality:float;adaptation:float;retention:float;efficiency:float;robustness:float;novelty:float
    samples:int=0;cost:float=0.;lineage:str=""
class Repertoire:
    def __init__(self,bins=5):self.bins=bins;self.cells={};self.novelty=[]
    def descriptor(self,e):
        # MAP-Elites-like behavioral niche, not parameter niche
        return (min(self.bins-1,int(e.adaptation*self.bins)),min(self.bins-1,int(e.retention*self.bins)),
                min(self.bins-1,int(e.efficiency*self.bins)),min(self.bins-1,int(e.robustness*self.bins)))
    def fitness(self,e):return .4*e.quality+.2*e.adaptation+.2*e.retention+.1*e.efficiency+.1*e.robustness
    def consider(self,name,e):
        cell=self.descriptor(e);old=self.cells.get(cell)
        accepted=old is None or self.fitness(e)>self.fitness(old[1])
        if accepted:self.cells[cell]=(name,e)
        if e.novelty>=.6:self.novelty.append((name,e))
        return accepted
    def best_for(self,weights):
        def score(item):
            _,e=item;return sum(weights.get(k,0)*getattr(e,k) for k in weights)
        return max(self.cells.values(),key=score) if self.cells else None
    def save(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"cells":{"-".join(map(str,k)):{"name":n,"evidence":asdict(e)} for k,(n,e) in self.cells.items()},
          "novelty":[{"name":n,"evidence":asdict(e)} for n,e in self.novelty]},indent=2))

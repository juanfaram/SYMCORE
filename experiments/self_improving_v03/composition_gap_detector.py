#!/usr/bin/env python3
"""Detect composition gaps: components work alone but their composition fails."""
from collections import defaultdict,deque
class CompositionGapDetector:
    def __init__(self,component_floor=.70,composition_floor=.60,min_obs=20,window=128):
        self.component_floor=component_floor;self.composition_floor=composition_floor;self.min_obs=min_obs
        self.hist=defaultdict(lambda:{"components":defaultdict(lambda:deque(maxlen=window)),"joint":deque(maxlen=window)})
    def observe(self,name,component_scores,joint_score):
        h=self.hist[name]
        for k,v in component_scores.items():h["components"][k].append(float(v))
        h["joint"].append(float(joint_score))
    def gaps(self):
        out=[]
        for name,h in self.hist.items():
            if len(h["joint"])<self.min_obs:continue
            means={k:sum(v)/len(v) for k,v in h["components"].items() if len(v)>=self.min_obs}
            if len(means)<2:continue
            joint=sum(h["joint"])/len(h["joint"])
            if min(means.values())>=self.component_floor and joint<self.composition_floor:
                out.append({"composition":name,"components":means,"joint":joint,
                            "deficit":min(means.values())-joint})
        return sorted(out,key=lambda x:x["deficit"],reverse=True)

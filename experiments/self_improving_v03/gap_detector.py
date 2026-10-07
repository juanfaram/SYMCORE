#!/usr/bin/env python3
"""Detect capability gaps: contexts where no existing expert demonstrates competence."""
from collections import defaultdict,deque
class GapDetector:
    def __init__(self,threshold=.65,min_obs=40,window=128):
        self.threshold=threshold;self.min_obs=min_obs;self.hist=defaultdict(lambda:defaultdict(lambda:deque(maxlen=window)))
    def observe(self,context,expert_scores):
        key=tuple(sorted((k,str(v)) for k,v in context.items() if k in ("task","domain","difficulty")))
        for expert,score in expert_scores.items():self.hist[key][expert].append(float(score))
    def gaps(self):
        out=[]
        for key,experts in self.hist.items():
            means={e:sum(xs)/len(xs) for e,xs in experts.items() if len(xs)>=self.min_obs}
            if means and max(means.values())<self.threshold:out.append({"context":dict(key),"best":max(means.values()),"experts":means})
        return sorted(out,key=lambda x:x["best"])

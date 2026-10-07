#!/usr/bin/env python3
"""Representation-aware skill learner: transfers structure, not previous task labels."""
from collections import defaultdict
class RepresentationLearner:
    def __init__(self,actions,representation):
        self.actions=actions;self.rep=representation;self.stats=defaultdict(lambda:defaultdict(lambda:[0,0.0]))
    def choose(self,c):
        scores={a:0. for a in self.actions};weights={a:0. for a in self.actions}
        for factor,value,w in self.rep.keys(c):
            key=(factor,value)
            for a in self.actions:
                n,r=self.stats[key][a]
                if n:scores[a]+=w*r/n;weights[a]+=w
        return max(self.actions,key=lambda a:scores[a]/weights[a] if weights[a] else 0.)
    def feedback(self,a,c,reward):
        self.rep.observe(c,reward)
        for factor,value,w in self.rep.keys(c):
            s=self.stats[(factor,value)][a];s[0]+=1;s[1]+=float(reward)

#!/usr/bin/env python3
"""v1.1 non-interfering skill memory with compositional fallback."""
from collections import defaultdict
class SkillMemory:
    def __init__(self,actions):self.actions=actions;self.stats=defaultdict(lambda:defaultdict(lambda:[0,0.0]))
    def _keys(self,c):
        # exact + factorized representations support retention and zero-shot composition
        return [("exact",c.get("task"),c.get("domain"),c.get("difficulty")),
                ("domain",c.get("domain")),("difficulty",c.get("difficulty")),("task",c.get("task"))]
    def choose(self,c):
        scores={a:0. for a in self.actions};weights={a:0. for a in self.actions}
        for key in self._keys(c):
            for a in self.actions:
                n,r=self.stats[key][a]
                if n:scores[a]+=r/n*min(1.,n/20);weights[a]+=min(1.,n/20)
        return max(self.actions,key=lambda a:scores[a]/weights[a] if weights[a] else 0.)
    def feedback(self,a,c,reward):
        for key in self._keys(c):
            s=self.stats[key][a];s[0]+=1;s[1]+=float(reward)

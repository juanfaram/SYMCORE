#!/usr/bin/env python3
"""Hierarchical non-interfering skill memory with compositional fallback."""
from collections import defaultdict
class SkillMemory:
    def __init__(self,actions):
        self.actions=actions;self.stats=defaultdict(lambda:defaultdict(lambda:[0,0.0]))
    def _keys(self,c):
        return [("exact",c.get("task"),c.get("domain"),c.get("difficulty")),
                ("pair",c.get("domain"),c.get("difficulty")),
                ("difficulty",c.get("difficulty")),("domain",c.get("domain")),("task",c.get("task"))]
    def choose(self,c):
        # Specific evidence wins. Back off only when a level lacks support.
        levels=[self._keys(c)[:1],self._keys(c)[1:2],self._keys(c)[2:4],self._keys(c)[4:]]
        for keys in levels:
            score={a:[0.0,0] for a in self.actions}
            for key in keys:
                for a in self.actions:
                    n,r=self.stats[key][a]
                    if n:score[a][0]+=r;score[a][1]+=n
            supported=[a for a,(r,n) in score.items() if n>=5]
            if supported:return max(supported,key=lambda a:(score[a][0]/score[a][1],score[a][1]))
        return self.actions[0]
    def feedback(self,a,c,reward):
        for key in self._keys(c):
            s=self.stats[key][a];s[0]+=1;s[1]+=float(reward)

#!/usr/bin/env python3
"""v1.3 meta-transfer: learn reusable action priors across related tasks."""
from collections import defaultdict
class TransferPrior:
    def __init__(self,actions):self.actions=actions;self.factor=defaultdict(lambda:defaultdict(lambda:[0,0.0]))
    def observe(self,context,action,reward):
        for key in (("domain",context.get("domain")),("difficulty",context.get("difficulty"))):
            s=self.factor[key][action];s[0]+=1;s[1]+=float(reward)
    def prior(self,context):
        score={a:0. for a in self.actions};weight={a:0 for a in self.actions}
        for key in (("domain",context.get("domain")),("difficulty",context.get("difficulty"))):
            for a in self.actions:
                n,r=self.factor[key][a]
                if n:score[a]+=r/n;weight[a]+=1
        return {a:(score[a]/weight[a] if weight[a] else 0.) for a in self.actions}
    def seed(self,memory,context,strength=8):
        p=self.prior(context)
        for a,v in p.items():
            if v>0:
                for key in memory._keys(context):
                    s=memory.stats[key][a];s[0]+=strength;s[1]+=strength*v

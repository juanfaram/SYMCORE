#!/usr/bin/env python3
"""Learn candidate latent factors from interaction traces instead of a fixed semantic vocabulary."""
from collections import defaultdict
class LatentFactorMiner:
    def __init__(self,min_support=30,min_gain=.15):
        self.min_support=min_support;self.min_gain=min_gain;self.rows=[]
    def observe(self,raw_features,success):
        self.rows.append((dict(raw_features),float(success)))
    def discover(self):
        if not self.rows:return []
        base=sum(y for _,y in self.rows)/len(self.rows);factors=[]
        keys=sorted(set().union(*(x.keys() for x,_ in self.rows)))
        for k in keys:
            groups=defaultdict(list)
            for x,y in self.rows:groups[str(x.get(k))].append(y)
            vals=[sum(v)/len(v) for v in groups.values() if len(v)>=self.min_support]
            if len(vals)>=2:
                spread=max(vals)-min(vals)
                if spread>=self.min_gain:factors.append({"factor":k,"spread":spread,"groups":len(vals),"support":sum(len(v) for v in groups.values())})
        # pairwise conjunctions are genuinely proposed structure not present as single factors
        for i,k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                groups=defaultdict(list)
                for x,y in self.rows:groups[(str(x.get(k1)),str(x.get(k2)))].append(y)
                vals=[sum(v)/len(v) for v in groups.values() if len(v)>=self.min_support]
                if len(vals)>=2 and max(vals)-min(vals)>=self.min_gain:
                    factors.append({"factor":f"{k1}&{k2}","spread":max(vals)-min(vals),"groups":len(vals),"support":sum(len(v) for v in groups.values())})
        return sorted(factors,key=lambda z:z["spread"],reverse=True)

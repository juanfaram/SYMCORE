#!/usr/bin/env python3
"""Tabular capability factory: generate non-redundant feature specialists from observed error structure."""
import itertools
class TabularCapabilityFactory:
    def propose(self,n_features,feature_effects,max_size=4,top_k=6):
        ranked=sorted(range(n_features),key=lambda j:feature_effects.get(j,0),reverse=True)
        specs=[];seen=set()
        for size in range(1,min(max_size,len(ranked))+1):
            for comb in itertools.combinations(ranked[:min(6,len(ranked))],size):
                if comb not in seen:specs.append({"features":comb,"novelty":1/(1+size),"signal":sum(feature_effects.get(j,0) for j in comb)});seen.add(comb)
        return sorted(specs,key=lambda x:(x["signal"]+x["novelty"]),reverse=True)[:top_k]

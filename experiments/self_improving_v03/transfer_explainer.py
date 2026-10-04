#!/usr/bin/env python3
"""Explain directional transfer by shared discovered subspaces."""
def explain(source_trace,target_trace,miner):
    factors=miner.discover();shared=[]
    sk=set(source_trace);tk=set(target_trace)
    for f in factors:
        atoms=set(f["factor"].split("&"))
        if atoms<=sk and atoms<=tk:shared.append(f)
    return {"shared_subspace":shared[:10],"explained":bool(shared)}

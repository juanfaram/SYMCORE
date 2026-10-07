#!/usr/bin/env python3
"""Acceleration evidence contract: A>0 and dA/dE>0 require replicated acquisition-cost curves."""
import math,statistics
def acceleration(costs):
    if len(costs)<3:return {"A":0.,"dA_dE":0.,"passed_A":False,"passed_dA":False}
    # A: normalized reduction from first to last; dA/dE: acceleration of reductions
    A=(costs[0]-costs[-1])/max(costs[0],1e-9)
    gains=[costs[i-1]-costs[i] for i in range(1,len(costs))]
    dA=(gains[-1]-gains[0])/max(costs[0],1e-9) if len(gains)>1 else 0.
    return {"A":A,"dA_dE":dA,"passed_A":A>0,"passed_dA":dA>0,"costs":list(costs)}
def replicated(curves):
    rs=[acceleration(c) for c in curves]
    return {"runs":rs,"A_positive_fraction":sum(x["passed_A"] for x in rs)/len(rs),
            "dA_positive_fraction":sum(x["passed_dA"] for x in rs)/len(rs)}

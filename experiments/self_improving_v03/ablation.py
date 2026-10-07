#!/usr/bin/env python3
"""Causal ablation gate: a mechanism earns credit only if removing it measurably hurts."""
import statistics,math
def paired_ablation(full,ablated,min_effect=.01):
    if len(full)!=len(ablated) or len(full)<5:return {"supported":False,"reason":"insufficient_replication"}
    d=[a-f for f,a in zip(full,ablated)] # lower loss is better; positive means full helps
    mean=statistics.fmean(d);ci=1.96*statistics.stdev(d)/math.sqrt(len(d))
    return {"supported":mean-ci>min_effect,"effect":mean,"ci95":ci,"lower":mean-ci}

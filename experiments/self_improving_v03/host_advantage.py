#!/usr/bin/env python3
"""Universal host-advantage evaluator: paired causal evidence with confidence bounds."""
import math,statistics
def paired_advantage(control_losses,symcore_losses,min_gain=0.0):
    if len(control_losses)!=len(symcore_losses) or len(control_losses)<30:return {"passed":False,"reason":"insufficient_paired_data"}
    d=[c-s for c,s in zip(control_losses,symcore_losses)]
    mean=statistics.fmean(d);se=statistics.stdev(d)/math.sqrt(len(d));lower=mean-1.96*se
    rel=mean/max(1e-12,statistics.fmean(control_losses))
    return {"passed":lower>min_gain,"mean_gain":mean,"relative_gain":rel,"ci95":1.96*se,"lower":lower,"n":len(d)}

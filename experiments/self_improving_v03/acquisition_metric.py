#!/usr/bin/env python3
"""v2 acquisition cost: area-under-learning-curve + threshold latency + compute, avoiding floor saturation."""
import statistics
def acquisition_cost(losses,target=None,compute_per_step=1.0):
    if not losses:return {"cost":float("inf"),"auc":float("inf"),"latency":0}
    baseline=max(statistics.fmean(losses[:min(25,len(losses))]),1e-9)
    norm=[min(2.0,x/baseline) for x in losses]
    auc=sum(norm)
    if target is None:target=min(.7,statistics.fmean(norm[-min(25,len(norm)):])*.1+0.5)
    latency=len(norm)+1
    for i in range(20,len(norm)+1):
        if statistics.fmean(norm[i-20:i])<=target:latency=i;break
    # continuous cost avoids identical threshold-floor values
    cost=auc+0.25*latency+0.01*compute_per_step*len(losses)
    return {"cost":cost,"auc":auc,"latency":latency,"target":target}

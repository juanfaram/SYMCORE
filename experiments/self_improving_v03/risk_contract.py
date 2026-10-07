#!/usr/bin/env python3
"""North-star risk contract: estimate probability that SYMCORE exceeds an allowed regression envelope."""
import math,statistics
def risk_report(control,symcore,rmax=0.05):
    if len(control)!=len(symcore) or not control:return {"passed":False,"reason":"invalid"}
    regress=[(s-c)/max(abs(c),1e-9) for c,s in zip(control,symcore)]
    bad=sum(x>rmax for x in regress);n=len(regress);p=bad/n
    # Wilson upper 95% bound
    z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return {"n":n,"violations":bad,"p_hat":p,"p_upper95":min(1.,center+half),"rmax":rmax}
def passes(report,delta=.05):return report.get("p_upper95",1.)<delta

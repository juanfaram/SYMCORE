#!/usr/bin/env python3
"""Sensitivity audit for acquisition-cost v2."""
import json,math,random
from pathlib import Path
from acquisition_metric import acquisition_cost
def losses(seed,noise,period,n=500):
 r=random.Random(seed);mem={};out=[]
 for i in range(1,n+1):
  y=50+20*math.sin(2*math.pi*i/period)+r.gauss(0,noise);k=i%period;p=mem.get(k,50);out.append(abs(p-y));mem[k]=.8*mem.get(k,y)+.2*y
 return out
def run():
 levels={"easy":(2,24),"medium":(7,17),"hard":(14,11)};seeds=(3,11,29,47,83,131,197)
 vals={k:[acquisition_cost(losses(s,*v))["cost"] for s in seeds] for k,v in levels.items()}
 means={k:sum(v)/len(v) for k,v in vals.items()};sensitive=means["easy"]<means["medium"]<means["hard"]
 out={"costs":vals,"means":means,"sensitive":sensitive,"spread":max(means.values())-min(means.values())}
 Path("artifacts/instrument_sensitivity_v2.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

#!/usr/bin/env python3
"""Instrument sensitivity audit: can acquisition-cost metric detect known difficulty differences?"""
import json,math,random,statistics
from pathlib import Path
def acquire(seed,noise,period,threshold=12,n=1200):
 r=random.Random(seed);mem={};recent=[]
 for i in range(1,n+1):
  y=50+20*math.sin(2*math.pi*i/period)+r.gauss(0,noise);k=i%period;p=mem.get(k,y);recent.append(abs(p-y))
  if len(recent)>100:recent.pop(0)
  mem[k]=.75*mem.get(k,y)+.25*y
  if len(recent)==100 and statistics.fmean(recent)<=threshold:return i
 return n+1
def run():
 levels={"easy":(2,24),"medium":(6,18),"hard":(12,11)};seeds=(3,11,29,47,83,131,197)
 vals={k:[acquire(s,*v) for s in seeds] for k,v in levels.items()}
 means={k:statistics.fmean(v) for k,v in vals.items()}
 sensitive=means["easy"]<means["medium"]<means["hard"]
 spread=max(means.values())-min(means.values())
 out={"costs":vals,"means":means,"sensitive":sensitive,"spread":spread}
 Path("artifacts/instrument_sensitivity.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

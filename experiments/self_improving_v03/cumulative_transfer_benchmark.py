#!/usr/bin/env python3
"""Cumulative transfer: does X+Y reduce the cost of acquiring Z vs X alone?"""
import json,random,statistics
from pathlib import Path
from skill_memory import SkillMemory
from transfer_prior import TransferPrior
A=["fast","balanced","deep","creative"]
def context(r):return {"task":"solve","domain":r.choice(["code","analysis","writing"]),"difficulty":r.choice(["simple","normal","hard"])}
def x(c):return {"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]]
def y(c):return {"code":"deep","analysis":"balanced","writing":"creative"}[c["domain"]]
def z(c):return "creative" if c["domain"]=="writing" else ("deep" if c["difficulty"]=="hard" else "balanced")
def teach(m,p,fn,r,n=1800):
 for _ in range(n):
  c=context(r);a=fn(c);m.feedback(a,c,1);p.observe(c,a,1)
def acquire(m,p,r,max_steps=3000):
 recent=[]
 for i in range(1,max_steps+1):
  c=context(r);p.seed(m,c,2);a=m.choose(c);ok=a==z(c);rew=1 if ok else -.3
  m.feedback(a,c,rew);p.observe(c,a,rew);recent.append(ok)
  if len(recent)>100:recent.pop(0)
  if len(recent)==100 and sum(recent)>=75:return i
 return max_steps+1
def trial(seed,both):
 r=random.Random(seed);m=SkillMemory(A);p=TransferPrior(A);teach(m,p,x,r)
 if both:teach(m,p,y,r)
 return acquire(m,p,r)
def run(seeds=(13,31,59,89,127,163,211)):
 one=[trial(s,False) for s in seeds];two=[trial(s,True) for s in seeds]
 a=statistics.fmean(one);b=statistics.fmean(two);gain=(a-b)/max(1,a)
 out={"seeds":list(seeds),"X_then_Z":one,"X_Y_then_Z":two,"X_mean":a,"XY_mean":b,
      "cumulative_transfer":round(gain,4),"positive":gain>0}
 Path("artifacts/cumulative_transfer_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

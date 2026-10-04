#!/usr/bin/env python3
"""SYMCORE v0.2 - evolutionary online learner with drift-triggered mutation."""
from __future__ import annotations
import argparse,csv,json,math,random,urllib.request
from collections import deque
from dataclasses import dataclass,field
from pathlib import Path

DATA_URL="https://raw.githubusercontent.com/KeithJLZ/UCI-Bike-Sharing-Dataset/master/hour.csv"
WINDOW=500

def features(r,rich=False):
    hr=float(r["hr"]); dow=float(r["weekday"]); mn=float(r["mnth"])
    x={"bias":1.0,"hr_sin":math.sin(2*math.pi*hr/24),"hr_cos":math.cos(2*math.pi*hr/24),
       "dow_sin":math.sin(2*math.pi*dow/7),"dow_cos":math.cos(2*math.pi*dow/7),
       "month_sin":math.sin(2*math.pi*mn/12),"month_cos":math.cos(2*math.pi*mn/12),
       "temp":float(r["temp"]),"atemp":float(r["atemp"]),"hum":float(r["hum"]),
       "windspeed":float(r["windspeed"]),"workingday":float(r["workingday"]),
       "holiday":float(r["holiday"]),"weathersit":float(r["weathersit"])/4,"yr":float(r["yr"])}
    if rich:
        x.update({"rush_am":float(7<=hr<=9),"rush_pm":float(16<=hr<=19),
          "temp2":float(r["temp"])**2,"hum2":float(r["hum"])**2,
          "work_x_hr_sin":float(r["workingday"])*math.sin(2*math.pi*hr/24),
          "work_x_hr_cos":float(r["workingday"])*math.cos(2*math.pi*hr/24)})
    return x

@dataclass
class Model:
    lr:float; rich:bool; l2:float=0.0
    w:dict=field(default_factory=dict); g2:dict=field(default_factory=dict)
    def predict(self,x): return max(0.,sum(self.w.get(k,0.)*v for k,v in x.items()))
    def learn(self,x,y):
        err=max(-500.,min(500.,self.predict(x)-y))
        for k,v in x.items():
            grad=err*v+self.l2*self.w.get(k,0.)
            self.g2[k]=self.g2.get(k,0.)+grad*grad
            self.w[k]=self.w.get(k,0.)-self.lr*grad/(math.sqrt(self.g2[k])+1e-8)

@dataclass
class Candidate:
    name:str; model:Model; born:int=0; parent:str|None=None
    errors:deque=field(default_factory=lambda:deque(maxlen=WINDOW))
    total:float=0.; n:int=0
    def observe(self,r,y):
        x=features(r,self.model.rich); pred=self.model.predict(x); ae=abs(pred-y)
        self.errors.append(ae); self.total+=ae; self.n+=1; self.model.learn(x,y)
    @property
    def rolling(self): return sum(self.errors)/len(self.errors) if self.errors else float("inf")
    @property
    def mae(self): return self.total/self.n if self.n else float("inf")

def mutate(parent,idx,at,rng):
    lr=max(.005,min(1.,parent.model.lr*math.exp(rng.uniform(-.8,.8))))
    rich=parent.model.rich if rng.random()>.25 else not parent.model.rich
    l2=max(0.,min(.05,parent.model.l2+rng.choice([-1,1])*10**rng.uniform(-5,-2)))
    m=Model(lr,rich,l2)
    # inherit knowledge where features overlap, then continue adapting
    m.w=dict(parent.model.w); m.g2={k:max(1.,v*.5) for k,v in parent.model.g2.items()}
    return Candidate(f"gen{at}-{idx}",m,at,parent.name)

class DriftDetector:
    """Dependency-free Page-Hinkley-style detector over champion absolute error."""
    def __init__(self,delta=.05,threshold=120.,warmup=200):
        self.delta=delta; self.threshold=threshold; self.warmup=warmup; self.n=0
        self.mean=0.; self.cum=0.; self.min_cum=0.
    def update(self,x):
        self.n+=1; self.mean+=(x-self.mean)/self.n
        self.cum+=x-self.mean-self.delta; self.min_cum=min(self.min_cum,self.cum)
        drift=self.n>=self.warmup and self.cum-self.min_cum>self.threshold
        if drift: self.__init__(self.delta,self.threshold,self.warmup)
        return drift

def ensure_data(path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve(DATA_URL,path)

def run(path,interval=500,seed=7):
    rng=random.Random(seed)
    pool=[Candidate("seed",Model(.1,True))]
    champion=pool[0]; detector=DriftDetector(); events=[]; rows=0
    with path.open(newline="",encoding="utf-8") as f:
      for r in csv.DictReader(f):
        rows+=1; y=float(r["cnt"])
        # detect regime changes using pre-update champion error
        cx=features(r,champion.model.rich); cerr=abs(champion.model.predict(cx)-y)
        drift=detector.update(cerr)
        for c in pool: c.observe(r,y)
        evolve=(rows>=WINDOW and rows%interval==0) or drift
        if evolve:
            reason="drift" if drift else "scheduled"
            mature=[c for c in pool if len(c.errors)>=min(WINDOW,max(100,rows-c.born))]
            best=min(mature,key=lambda c:c.rolling)
            # require 2% recent improvement before replacing champion
            if best is not champion and best.rolling < champion.rolling*.98:
                events.append({"at":rows,"type":"promotion","reason":reason,"from":champion.name,
                  "to":best.name,"old_mae":round(champion.rolling,3),"new_mae":round(best.rolling,3)})
                champion=best
            children=[mutate(champion,i,rows,rng) for i in range(3)]
            events.append({"at":rows,"type":"mutation","reason":reason,"parent":champion.name,
              "children":[{"name":c.name,"lr":round(c.model.lr,5),"rich":c.model.rich,
                           "l2":round(c.model.l2,7)} for c in children]})
            # bounded population: champion + strongest recent challengers + new mutations
            keep=sorted([c for c in pool if c is not champion],key=lambda c:c.rolling)[:3]
            pool=[champion]+keep+children
            print(f"{rows:5d} {reason:9s} champion={champion.name:12s} MAE={champion.rolling:7.2f} pool={len(pool)}")
    result={"version":"0.2","rows":rows,"champion":champion.name,
      "champion_rolling_mae":round(champion.rolling,3),"events":events,
      "population":sorted([{"name":c.name,"parent":c.parent,"born":c.born,
        "lr":round(c.model.lr,6),"rich":c.model.rich,"l2":round(c.model.l2,8),
        "mae":round(c.mae,3),"rolling_mae":round(c.rolling,3)} for c in pool],
        key=lambda z:z["rolling_mae"])}
    Path("evolution.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/hour.csv")
    p.add_argument("--interval",type=int,default=500); p.add_argument("--seed",type=int,default=7)
    a=p.parse_args(); path=Path(a.data); ensure_data(path); run(path,a.interval,a.seed)

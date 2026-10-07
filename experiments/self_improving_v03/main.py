#!/usr/bin/env python3
"""SYMCORE v0.3: bounded evolutionary online-learning engine."""
from __future__ import annotations
import argparse,csv,json,math,pickle,random,statistics,urllib.request
from collections import deque
from dataclasses import dataclass,field,asdict
from pathlib import Path

DATA_URL="https://raw.githubusercontent.com/KeithJLZ/UCI-Bike-Sharing-Dataset/master/hour.csv"

@dataclass(frozen=True)
class Genome:
    lr:float=.1; l2:float=0.; rich:bool=True; momentum:float=0.; clip:float=500.
    def mutate(self,rng):
        return Genome(max(.002,min(1.,self.lr*math.exp(rng.uniform(-.7,.7)))),
          max(0.,min(.05,self.l2+rng.choice((-1,1))*10**rng.uniform(-6,-2))),
          self.rich if rng.random()>.2 else not self.rich,
          max(0.,min(.95,self.momentum+rng.uniform(-.2,.2))),
          max(50.,min(1000.,self.clip*math.exp(rng.uniform(-.25,.25)))))

def feats(r,rich):
    h=float(r["hr"]); d=float(r["weekday"]); m=float(r["mnth"])
    x={"b":1.,"hs":math.sin(2*math.pi*h/24),"hc":math.cos(2*math.pi*h/24),
       "ds":math.sin(2*math.pi*d/7),"dc":math.cos(2*math.pi*d/7),
       "ms":math.sin(2*math.pi*m/12),"mc":math.cos(2*math.pi*m/12),
       "temp":float(r["temp"]),"atemp":float(r["atemp"]),"hum":float(r["hum"]),
       "wind":float(r["windspeed"]),"work":float(r["workingday"]),
       "holiday":float(r["holiday"]),"weather":float(r["weathersit"])/4,"year":float(r["yr"])}
    if rich:x.update({"am":float(7<=h<=9),"pm":float(16<=h<=19),"t2":x["temp"]**2,
      "h2":x["hum"]**2,"whs":x["work"]*x["hs"],"whc":x["work"]*x["hc"],
      "temp_hs":x["temp"]*x["hs"],"temp_hc":x["temp"]*x["hc"]})
    return x

@dataclass
class Learner:
    genome:Genome; w:dict=field(default_factory=dict); g2:dict=field(default_factory=dict); vel:dict=field(default_factory=dict)
    def predict(self,x):return max(0.,sum(self.w.get(k,0.)*v for k,v in x.items()))
    def learn(self,x,y):
        e=max(-self.genome.clip,min(self.genome.clip,self.predict(x)-y))
        for k,v in x.items():
            g=e*v+self.genome.l2*self.w.get(k,0.); self.g2[k]=self.g2.get(k,0.)+g*g
            step=self.genome.lr*g/(math.sqrt(self.g2[k])+1e-8)
            self.vel[k]=self.genome.momentum*self.vel.get(k,0.)+step
            self.w[k]=self.w.get(k,0.)-self.vel[k]

@dataclass
class Agent:
    id:str; learner:Learner; born:int=0; parent:str|None=None
    err:deque=field(default_factory=lambda:deque(maxlen=512)); total:float=0.; n:int=0
    def step(self,r,y):
        x=feats(r,self.learner.genome.rich); p=self.learner.predict(x); ae=abs(p-y)
        self.err.append(ae);self.total+=ae;self.n+=1;self.learner.learn(x,y);return p,ae
    @property
    def score(self):return sum(self.err)/len(self.err) if self.err else float("inf")
    def child(self,g,id_,at):
        l=Learner(g,dict(self.learner.w),{k:max(1.,v*.5) for k,v in self.learner.g2.items()},dict(self.learner.vel))
        return Agent(id_,l,at,self.id)

class PageHinkley:
    def __init__(self,threshold=180.):self.threshold=threshold;self.reset()
    def reset(self):self.n=0;self.mean=0.;self.cum=0.;self.low=0.
    def update(self,x):
        self.n+=1;self.mean+=(x-self.mean)/self.n;self.cum+=x-self.mean-.05;self.low=min(self.low,self.cum)
        hit=self.n>200 and self.cum-self.low>self.threshold
        if hit:self.reset()
        return hit

def paired_gate(champ,challenger,min_gain=.015):
    n=min(len(champ.err),len(challenger.err))
    if n<128:return False,0.
    a=list(champ.err)[-n:];b=list(challenger.err)[-n:];diff=[x-y for x,y in zip(a,b)]
    gain=(statistics.mean(a)-statistics.mean(b))/max(statistics.mean(a),1e-9)
    # conservative normal approximation on paired errors
    sd=statistics.stdev(diff) if n>1 else 0.; se=sd/math.sqrt(n) if sd else 0.
    lower=statistics.mean(diff)-1.96*se
    return gain>=min_gain and lower>0,gain

def ensure(path):
    if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);urllib.request.urlretrieve(DATA_URL,path)

def run(path,out,interval=512,pop_size=10,children=5,seed=17,checkpoint_every=2048):
    rng=random.Random(seed);champ=Agent("root",Learner(Genome()));pool=[champ];ph=PageHinkley()
    events=[];genealogy=[];rows=0
    with path.open(newline="",encoding="utf-8") as f:
      for r in csv.DictReader(f):
        rows+=1;y=float(r["cnt"]);x=feats(r,champ.learner.genome.rich)
        drift=ph.update(abs(champ.learner.predict(x)-y))
        for a in pool:a.step(r,y)
        if rows>=512 and (rows%interval==0 or drift):
            ranked=sorted(pool,key=lambda a:a.score)
            for challenger in ranked:
                if challenger is champ:continue
                ok,gain=paired_gate(champ,challenger)
                if ok:
                    events.append({"at":rows,"type":"promotion","from":champ.id,"to":challenger.id,"gain":round(gain,4)})
                    champ=challenger;break
            newborn=[]
            for i in range(children):
                g=champ.learner.genome.mutate(rng);c=champ.child(g,f"g{rows}-{i}",rows)
                newborn.append(c);genealogy.append({"child":c.id,"parent":champ.id,"at":rows,"genome":asdict(g)})
            survivors=[a for a in ranked if a is not champ][:max(0,pop_size-1-children)]
            pool=[champ]+survivors+newborn
            events.append({"at":rows,"type":"evolution","trigger":"drift" if drift else "scheduled",
                           "champion":champ.id,"population":len(pool)})
        if checkpoint_every and rows%checkpoint_every==0:
            out.mkdir(parents=True,exist_ok=True)
            with (out/"checkpoint.pkl").open("wb") as z:pickle.dump({"row":rows,"champion":champ,"pool":pool},z)
    ranked=sorted(pool,key=lambda a:a.score)
    # ensemble estimate from final population's current recent MAE; members exposed for next live stage
    report={"version":"0.3","rows":rows,"champion":champ.id,"champion_mae":round(champ.score,3),
      "champion_genome":asdict(champ.learner.genome),"population":[{"id":a.id,"parent":a.parent,
      "born":a.born,"mae":round(a.score,3),"genome":asdict(a.learner.genome)} for a in ranked],
      "ensemble_members":[a.id for a in ranked[:3]],"events":events,"genealogy":genealogy}
    out.mkdir(parents=True,exist_ok=True);(out/"evolution.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("version","rows","champion","champion_mae","champion_genome","ensemble_members")},indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/hour.csv");p.add_argument("--out",default="artifacts")
    p.add_argument("--interval",type=int,default=512);p.add_argument("--population",type=int,default=10)
    p.add_argument("--children",type=int,default=5);p.add_argument("--seed",type=int,default=17)
    a=p.parse_args();path=Path(a.data);ensure(path);run(path,Path(a.out),a.interval,a.population,a.children,a.seed)

#!/usr/bin/env python3
"""SYMCORE v0.1 - online self-improving model selection on real bike demand data."""
from __future__ import annotations
import argparse, csv, json, math, urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/KeithJLZ/UCI-Bike-Sharing-Dataset/master/hour.csv"

def features(r, rich=False):
    hr=float(r["hr"]); dow=float(r["weekday"]); mn=float(r["mnth"])
    x={
      "bias":1.0,
      "hr_sin":math.sin(2*math.pi*hr/24), "hr_cos":math.cos(2*math.pi*hr/24),
      "dow_sin":math.sin(2*math.pi*dow/7), "dow_cos":math.cos(2*math.pi*dow/7),
      "month_sin":math.sin(2*math.pi*mn/12), "month_cos":math.cos(2*math.pi*mn/12),
      "temp":float(r["temp"]), "atemp":float(r["atemp"]), "hum":float(r["hum"]),
      "windspeed":float(r["windspeed"]), "workingday":float(r["workingday"]),
      "holiday":float(r["holiday"]), "weathersit":float(r["weathersit"])/4.0,
      "yr":float(r["yr"]),
    }
    if rich:
        x.update({
          "rush_am": 1.0 if 7 <= hr <= 9 else 0.0,
          "rush_pm": 1.0 if 16 <= hr <= 19 else 0.0,
          "temp2": float(r["temp"])**2,
          "hum2": float(r["hum"])**2,
          "work_x_hr_sin": float(r["workingday"])*math.sin(2*math.pi*hr/24),
          "work_x_hr_cos": float(r["workingday"])*math.cos(2*math.pi*hr/24),
        })
    return x

@dataclass
class OnlineRegressor:
    lr: float
    rich: bool
    w: dict = field(default_factory=dict)
    g2: dict = field(default_factory=dict)
    def predict(self, x):
        return max(0.0, sum(self.w.get(k,0.0)*v for k,v in x.items()))
    def learn(self, x, y):
        pred=self.predict(x)
        err=max(-500.0,min(500.0,pred-y))
        for k,v in x.items():
            grad=err*v
            self.g2[k]=self.g2.get(k,0.0)+grad*grad
            step=self.lr/(math.sqrt(self.g2[k])+1e-8)
            self.w[k]=self.w.get(k,0.0)-step*grad
        return pred

@dataclass
class Candidate:
    name: str
    model: OnlineRegressor
    errors: deque = field(default_factory=lambda: deque(maxlen=500))
    total_abs: float = 0.0
    n: int = 0
    def observe(self, r, y):
        x=features(r,self.model.rich)
        pred=self.model.predict(x)
        ae=abs(pred-y)
        self.errors.append(ae); self.total_abs += ae; self.n += 1
        self.model.learn(x,y)
        return pred,ae
    @property
    def rolling_mae(self):
        return sum(self.errors)/len(self.errors) if self.errors else float("inf")
    @property
    def mae(self):
        return self.total_abs/self.n if self.n else float("inf")

def ensure_data(path: Path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public dataset -> {path}")
    urllib.request.urlretrieve(DATA_URL,path)

def run(path, report_every=500):
    candidates=[
      Candidate("base-lr0.03",OnlineRegressor(.03,False)),
      Candidate("base-lr0.10",OnlineRegressor(.10,False)),
      Candidate("rich-lr0.03",OnlineRegressor(.03,True)),
      Candidate("rich-lr0.10",OnlineRegressor(.10,True)),
      Candidate("rich-lr0.30",OnlineRegressor(.30,True)),
    ]
    champion=candidates[0].name; switches=[]; rows=0
    with path.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y=float(r["cnt"]); rows+=1
            for c in candidates: c.observe(r,y)
            if rows >= 500 and rows % report_every == 0:
                best=min(candidates,key=lambda c:c.rolling_mae)
                if best.name != champion:
                    switches.append({"at":rows,"from":champion,"to":best.name,
                                     "rolling_mae":round(best.rolling_mae,3)})
                    champion=best.name
                print(f"{rows:5d} | champion={champion:14s} | rolling MAE={best.rolling_mae:8.2f}")
    best=min(candidates,key=lambda c:c.rolling_mae)
    result={
      "rows":rows,"champion":best.name,"champion_rolling_mae":round(best.rolling_mae,3),
      "switches":switches,
      "candidates":sorted([
        {"name":c.name,"mae":round(c.mae,3),"rolling_mae":round(c.rolling_mae,3),
         "features":"rich" if c.model.rich else "base","learning_rate":c.model.lr}
        for c in candidates],key=lambda z:z["rolling_mae"])
    }
    print("\n"+json.dumps(result,indent=2))
    Path("evolution.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/hour.csv")
    p.add_argument("--report-every",type=int,default=500)
    a=p.parse_args(); path=Path(a.data); ensure_data(path); run(path,a.report_every)

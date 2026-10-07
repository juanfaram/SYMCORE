#!/usr/bin/env python3
"""520 falsification: transfer evolutionary decisions through anonymous functional geometry."""
import json,math,random
from pathlib import Path
from functional_geometry import FunctionalGeometry,distance
OPS=("adapt","stabilize","compress","noop")
def trace(kind,seed,n=64):
 r=random.Random(seed);rows=[]
 for t in range(n):
  if kind=="change":base=[.01*t+(1 if t>44 else 0),.4+(.5 if t>44 else 0),.5+r.gauss(0,.12)]
  elif kind=="noisy":base=[.5+r.gauss(0,.5),.5+r.gauss(0,.45),.8+r.gauss(0,.25)]
  elif kind=="redundant":base=[.35+.03*math.sin(t/5),.36+.03*math.sin(t/5),.9]
  else:base=[.3+.02*math.sin(t/8),.4+.02*math.cos(t/9),.25]
  rows.append(base)
 return rows
def encode(kind,seed,host):
 g=FunctionalGeometry()
 names=("error","uncertainty","cost") if host=="A" else ("sensor_z","sensor_q","sensor_m")
 scales=(1,1,1) if host=="A" else (7,.25,3.5);offs=(0,0,0) if host=="A" else (11,-8,40)
 for row in trace(kind,seed):
  g.observe({names[i]:offs[i]+scales[i]*row[i] for i in range(3)})
 return g.vector()
def target(kind):return {"change":"adapt","noisy":"stabilize","redundant":"compress","stable":"noop"}[kind]
def run(seeds=range(100)):
 kinds=("change","noisy","redundant","stable")
 train=[(encode(k,s,"A"),target(k)) for k in kinds for s in range(1000,1100)]
 rows={};passed=True
 for k in kinds:
  hit=0
  for s in seeds:
   z=encode(k,s,"B");near=sorted(train,key=lambda q:distance(z,q[0]))[:15]
   votes={o:sum(1 for _,y in near if y==o) for o in OPS};pred=max(votes,key=votes.get);hit+=pred==target(k)
  p=hit/len(seeds);zv=1.96;d=1+zv*zv/len(seeds);c=(p+zv*zv/(2*len(seeds)))/d;h=zv*math.sqrt(p*(1-p)/len(seeds)+zv*zv/(4*len(seeds)**2))/d;lo=c-h
  ok=p>=.60 and lo>.50;passed &= ok;rows[k]={"accuracy":p,"ci95_lower":lo,"passed":ok}
 out={"schema":"symcore.functional-geometry.v1","train_host":"A","test_host":"B-renamed-reordered-rescaled","states":rows,"passed":bool(passed)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/functional_geometry_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

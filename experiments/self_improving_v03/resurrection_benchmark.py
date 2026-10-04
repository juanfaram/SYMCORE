#!/usr/bin/env python3
"""Skill resurrection + compositional OOD benchmark for v1.1."""
import json,random
from pathlib import Path
from skill_memory import SkillMemory
def run(seed=515):
 rng=random.Random(seed);m=SkillMemory(["fast","balanced","deep","creative"])
 ds=["code","analysis","writing","forecast"];qs=["simple","normal","hard"]
 def difficulty(c):return {"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]]
 def domain(c):return {"code":"deep","analysis":"balanced","writing":"creative","forecast":"balanced"}[c["domain"]]
 def hybrid(c):return "creative" if c["domain"]=="writing" else ("fast" if c["difficulty"]=="simple" else "deep")
 skills=[difficulty,domain,hybrid];names=["difficulty","domain","hybrid"];probes=[]
 for fn in skills:
  for _ in range(3500):
   c={"task":"solve","domain":rng.choice(ds),"difficulty":rng.choice(qs)};a=fn(c);m.feedback(a,c,1)
  probes.append([{"task":"solve","domain":rng.choice(ds),"difficulty":rng.choice(qs)} for _ in range(800)])
 retention=[]
 for name,fn,p in zip(names,skills,probes):retention.append({"skill":name,"accuracy":round(sum(m.choose(c)==fn(c) for c in p)/len(p),4)})
 held={("writing","hard"),("forecast","simple"),("code","normal")}
 unseen=[{"task":"solve","domain":d,"difficulty":q} for d,q in held for _ in range(200)]
 # hybrid is compositional target
 ood=sum(m.choose(c)==hybrid(c) for c in unseen)/len(unseen)
 out={"resurrection":retention,"compositional_ood_accuracy":round(ood,4)}
 Path("artifacts/resurrection_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

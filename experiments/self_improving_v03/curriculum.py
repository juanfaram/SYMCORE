import json,random
from pathlib import Path
from interaction_learner import InteractionLearner
def acc(a,cases,target):return sum(a.choose(c)==target(c) for c in cases)/len(cases)
def run(seed=91):
 rng=random.Random(seed);a=InteractionLearner(["fast","balanced","deep","creative"],.06,seed,"artifacts/curriculum.jsonl")
 phases=[("difficulty",lambda c:{"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]]),("domain",lambda c:{"code":"deep","forecast":"balanced","writing":"creative","analysis":"deep"}[c["domain"]]),("hybrid",lambda c:"creative" if c["domain"]=="writing" else ("fast" if c["difficulty"]=="simple" else "deep"))]
 ds=["code","forecast","writing","analysis"];dfs=["simple","normal","hard"];hist=[];old=[]
 for name,target in phases:
  cases=[{"task":"solve","domain":rng.choice(ds),"difficulty":rng.choice(dfs)} for _ in range(3500)]
  for c in cases:
   x=a.choose(c);a.feedback(x,c,1 if x==target(c) else -.3,{"phase":name})
  probe=[{"task":"solve","domain":rng.choice(ds),"difficulty":rng.choice(dfs)} for _ in range(800)]
  hist.append({"phase":name,"current_accuracy":round(acc(a,probe,target),4),"retention":[{"skill":n,"accuracy":round(acc(a,p,t),4)} for n,t,p in old]});old.append((name,target,probe))
 out={"phases":hist};Path("artifacts/curriculum_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

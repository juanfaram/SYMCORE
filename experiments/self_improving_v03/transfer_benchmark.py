#!/usr/bin/env python3
"""Forward-transfer experiment: compare learning a target skill from scratch vs after prerequisites."""
import json,random
from pathlib import Path
from skill_memory import SkillMemory
def train_until(agent,target,gen,max_steps=5000,window=100,threshold=.8):
    recent=[]
    for i in range(1,max_steps+1):
        c=gen();a=agent.choose(c);ok=a==target(c);agent.feedback(a,c,1 if ok else -.3);recent.append(ok)
        if len(recent)>window:recent.pop(0)
        if len(recent)==window and sum(recent)/window>=threshold:return i
    return max_steps+1
def run(seed=808):
 def setup(rng):return lambda:{"task":"solve","domain":rng.choice(["code","writing","analysis"]),"difficulty":rng.choice(["simple","normal","hard"])}
 def prereq(c):return {"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]]
 def target(c):return "creative" if c["domain"]=="writing" else ("deep" if c["difficulty"]=="hard" else "balanced")
 r1=random.Random(seed);scratch=SkillMemory(["fast","balanced","deep","creative"]);ls=train_until(scratch,target,setup(r1))
 r2=random.Random(seed);experienced=SkillMemory(["fast","balanced","deep","creative"])
 for _ in range(3000):
  c=setup(r2)();a=prereq(c);experienced.feedback(a,c,1)
 le=train_until(experienced,target,setup(r2));ft=(ls-le)/max(1,ls)
 out={"scratch_latency":ls,"experienced_latency":le,"forward_transfer":round(ft,4)}
 Path("artifacts/transfer_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

from skill_memory import SkillMemory
from guarded_router import GuardedRouter
from temporal_metrics import adaptation_latency,forward_transfer
class E:
 def __init__(self,p):self.p=p
 def predict(self,r):return self.p
 def learn(self,r,y):pass
def test_router_falls_back_to_anchor():
 r=GuardedRouter({"anchor":E(10),"bad":E(100)},"anchor")
 for _ in range(100):p,c=r.learn({},10,"x")
 assert c=="anchor"
def test_skill_memory_retains_factorized_skills():
 m=SkillMemory(["fast","deep"])
 for _ in range(100):m.feedback("fast",{"task":"x","domain":"a","difficulty":"simple"},1)
 for _ in range(100):m.feedback("deep",{"task":"x","domain":"b","difficulty":"hard"},1)
 assert m.choose({"task":"x","domain":"new","difficulty":"simple"})=="fast"
 assert m.choose({"task":"x","domain":"new","difficulty":"hard"})=="deep"
def test_temporal_transfer_metric():
 assert adaptation_latency([False]*50+[True]*200,50,.8)>50
 assert forward_transfer(100,50)==.5

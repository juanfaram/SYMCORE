from mixture import AdaptiveMixture,ReplayMemory
class E:
 def __init__(self,p):self.p=p
 def predict(self,r):return self.p
 def learn(self,r,y):pass
def test_mixture_learns_best_expert():
 m=AdaptiveMixture({"bad":E(100),"good":E(10)},eta=.1)
 for _ in range(100):m.learn({},10,"x")
 assert m.weights("x")["good"]>.95
def test_replay_bounded():
 import random
 r=ReplayMemory(10);g=random.Random(1)
 for i in range(1000):r.add(i,g)
 assert len(r.items)==10 and r.seen==1000

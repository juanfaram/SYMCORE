import csv,io
from main import Genome,Learner,Agent,paired_gate,feats
ROW={"hr":"8","weekday":"1","mnth":"5","temp":".5","atemp":".5","hum":".5","windspeed":".2",
"workingday":"1","holiday":"0","weathersit":"1","yr":"1"}
def test_features_and_learning():
    a=Agent("a",Learner(Genome()))
    before=a.learner.predict(feats(ROW,True))
    for _ in range(300):a.step(ROW,100.)
    after=a.learner.predict(feats(ROW,True))
    assert after>before
def test_mutation_bounds():
    import random
    g=Genome().mutate(random.Random(1))
    assert .002<=g.lr<=1 and 0<=g.momentum<=.95 and 50<=g.clip<=1000
def test_promotion_gate_rejects_tiny_sample():
    a=Agent("a",Learner(Genome()));b=Agent("b",Learner(Genome()))
    for _ in range(20):a.err.append(10);b.err.append(1)
    assert paired_gate(a,b)[0] is False

from capability_lifecycle import CapabilityLifecycle,ReplicatedEvidence
from ablation import paired_ablation
def E(name,q=.8,a=.8,r=.8,e=.8,rob=.8,nov=.7,n=7):
 return ReplicatedEvidence(name,[q]*n,[a]*n,[r]*n,[e]*n,[rob]*n,nov,1.)
def test_lifecycle_requires_replication():
 x=CapabilityLifecycle();assert not x.submit(E("x",n=3))["validated"]
def test_lifecycle_validates_replicated_capability():
 x=CapabilityLifecycle();assert x.submit(E("x"))["validated"];assert len(x.repertoire.cells)==1
def test_ablation_requires_causal_effect():
 assert paired_ablation([1,1,1,1,1],[1.2,1.2,1.2,1.2,1.2])["supported"]
 assert not paired_ablation([1,1,1,1,1],[1.001,1.001,1.001,1.001,1.001])["supported"]

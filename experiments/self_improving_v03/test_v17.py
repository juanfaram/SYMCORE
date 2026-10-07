from latent_factor_miner import LatentFactorMiner
from transfer_explainer import explain
def test_miner_can_propose_conjunction():
 m=LatentFactorMiner(min_support=2,min_gain=.4)
 for a in (0,1):
  for b in (0,1):
   for _ in range(3):m.observe({"f1":a,"f2":b},1 if a==b else 0)
 fs=m.discover();assert any("&" in x["factor"] for x in fs)
def test_explainer_returns_shared_subspace():
 m=LatentFactorMiner(2,.4)
 for a in (0,1):
  for b in (0,1):
   for _ in range(3):m.observe({"x":a,"y":b},1 if a==b else 0)
 assert explain({"x","y"},{"x","y"},m)["explained"]

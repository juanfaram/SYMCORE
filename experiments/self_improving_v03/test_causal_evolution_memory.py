from causal_evolution_memory import CausalEvolutionMemory
def test_prediction_is_vector_not_scalar_and_uncertainty_shrinks_with_support():
 z=(.2,.3);e=(1.,0.);m=CausalEvolutionMemory(k=50)
 for i in range(4):m.observe(z,e,(.1,.2,.0,.05,.08),.4)
 p1=m.predict(z,e)
 for i in range(30):m.observe(z,e,(.1,.2,.0,.05,.08),.4)
 p2=m.predict(z,e)
 assert set(p2.means)=={"quality","adaptation","retention","efficiency","learnability"}
 assert max(p2.ci95.values())<=max(p1.ci95.values())+1e-12
def test_confident_regression_blocks_effect():
 m=CausalEvolutionMemory(k=20);z=(.5,.5)
 good=(1.,0.);bad=(0.,1.)
 for _ in range(20):
  m.observe(z,good,(.05,.04,.03,.02,.04),.5)
  m.observe(z,bad,(.5,.5,-.2,.5,.5),.2)
 choice,_=m.choose(z,{"good":good,"bad":bad},risk_budget=.05)
 assert choice=="good"

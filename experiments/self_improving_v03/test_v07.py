from capability_frontier import Capability,Frontier
def test_frontier_rejects_regressive_candidate():
 f=Frontier();assert f.consider("a",Capability(.8,.8,.8,.8))
 assert not f.consider("b",Capability(.7,.9,.8,.8))
def test_frontier_accepts_real_expansion():
 f=Frontier();f.consider("a",Capability(.8,.8,.8,.8))
 assert f.consider("b",Capability(.81,.85,.8,.8))

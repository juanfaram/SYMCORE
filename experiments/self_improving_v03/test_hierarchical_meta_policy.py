from hierarchical_meta_policy import HierarchicalMetaPolicy
class P:
 def __init__(self,m,ci=.001,c=.2):self.means={a:m[i] for i,a in enumerate(("quality","adaptation","retention","efficiency","learnability"))};self.ci95={a:ci for a in self.means};self.cost_mean=c;self.cost_ci95=.001
class M:
 def predict(self,z,e):return P(e)
def test_policy_selects_parameter_not_only_operation():
 p=HierarchicalMetaPolicy(M())
 q=p.choose((0,),{"create":[((.5,),(.02,.02,.02,.02,.02)),((1.,),(.08,.08,.08,.08,.08))],"noop":[((),(0,0,0,0,0))]})
 assert q.operation=="create" and q.params==(1.,)
def test_noop_beats_unsafe_high_gain():
 p=HierarchicalMetaPolicy(M(),risk_budget=.05)
 q=p.choose((0,),{"risky":[((1.,),(.9,.9,-.2,.9,.9))],"noop":[((),(0,0,0,0,0))]})
 assert q.operation=="noop"

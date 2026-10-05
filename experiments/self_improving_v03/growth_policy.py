from collections import defaultdict
class GrowthPolicy:
 def __init__(self,min_support=20):self.min_support=min_support;self.stats=defaultdict(lambda:[0,0.,0])
 def key(self,s):return tuple(int(s[k]>v) for k,v in (("error_velocity",0),("change_probability",.6),("recent_regret",0),("expert_disagreement",0)))
 def observe(self,state,future_gain,risk_violation=False):
  z=self.stats[self.key(state)];z[0]+=1;z[1]+=float(future_gain);z[2]+=int(risk_violation)
 def decide(self,state,risk_budget=.05):
  n,g,b=self.stats[self.key(state)];return n>=self.min_support and g/n>0 and b/n<risk_budget

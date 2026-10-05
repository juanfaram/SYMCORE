"""Policy over universal effect intents conditioned on acquisition-process state."""
from collections import defaultdict
from intervention_effects import INTENTS
class SemanticGrowthPolicy:
 def __init__(self,min_support=5):self.min_support=min_support;self.stats=defaultdict(lambda:[0,0.])
 def state_key(self,s):
  return (int(s.success_rate>=.5),int(s.policy_entropy>=1.),int(s.posterior_margin>=.2),
          int(s.recent_learning_velocity>=0),int(s.historical_transfer>=0))
 def observe(self,state,intent,utility):
  z=self.stats[(self.state_key(state),intent)];z[0]+=1;z[1]+=float(utility)
 def choose(self,state):
  key=self.state_key(state);seen=[(self.stats[(key,i)][1]/self.stats[(key,i)][0],i) for i in INTENTS if self.stats[(key,i)][0]>=self.min_support]
  return max(seen)[1] if seen else "balanced_adapt"

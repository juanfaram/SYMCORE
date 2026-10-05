"""Internal acquisition-process state with explicit cross-host observability mask."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class AcquisitionState:
 trials:int
 success_rate:float
 policy_entropy:float
 posterior_margin:float
 recent_learning_velocity:float
 historical_transfer:float
 observed:tuple=(True,True,True,True,True,True)
 def vector(self):
  vals=(self.trials,self.success_rate,self.policy_entropy,self.posterior_margin,self.recent_learning_velocity,self.historical_transfer)
  return tuple(v if m else math.nan for v,m in zip(vals,self.observed))
 def masked(self):
  vals=self.vector()
  return tuple(0.0 if isinstance(v,float) and math.isnan(v) else v for v in vals),self.observed

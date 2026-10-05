"""HostState v2 candidate: internal acquisition-process state, not world state."""
from dataclasses import dataclass
@dataclass(frozen=True)
class AcquisitionState:
 trials:int
 success_rate:float
 policy_entropy:float
 posterior_margin:float
 recent_learning_velocity:float
 historical_transfer:float
 def vector(self):return (self.trials,self.success_rate,self.policy_entropy,self.posterior_margin,self.recent_learning_velocity,self.historical_transfer)

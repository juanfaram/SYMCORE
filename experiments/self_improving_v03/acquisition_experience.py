"""Universal acquisition experience: transferable semantics, not local labels."""
from dataclasses import dataclass
from acquisition_state import AcquisitionState
from intervention_effects import Effect
@dataclass(frozen=True)
class AcquisitionExperience:
 before:AcquisitionState
 effect:Effect
 learning_delta:float
 cost:float
 risk:float
 confidence:float
 def utility(self,risk_weight=1.,cost_weight=.01):
  return self.learning_delta-risk_weight*self.risk-cost_weight*self.cost

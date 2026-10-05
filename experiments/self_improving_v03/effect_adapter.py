"""Bidirectional host adapter between local mechanisms and universal intervention effects."""
from intervention_effects import Effect,choose_local
class EffectAdapter:
 def __init__(self,local_effects):self.local_effects=dict(local_effects)
 def encode(self,local_name):return self.local_effects[local_name]
 def decode(self,intent):return choose_local(intent,self.local_effects)

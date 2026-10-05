#!/usr/bin/env python3
"""Bidirectional evolutionary engine: reversible structural operations gated by Frontier v3."""
from __future__ import annotations
from dataclasses import dataclass,field
from copy import deepcopy
from capability_frontier import Frontier,Capability
OPS=("create","modify","combine","freeze","prune","restore","noop")
@dataclass
class Component:
 name:str;payload:dict=field(default_factory=dict);active:bool=True
@dataclass
class EvolutionState:
 components:dict=field(default_factory=dict);archive:dict=field(default_factory=dict);version:int=0
 def clone(self):return deepcopy(self)
class BidirectionalEngine:
 def __init__(self,frontier=None):self.frontier=frontier or Frontier();self.history=[]
 def propose(self,state,op,target=None,payload=None):
  if op not in OPS:raise ValueError(op)
  return {"op":op,"target":target,"payload":payload or {},"base_version":state.version}
 def apply_shadow(self,state,proposal):
  s=state.clone();op=proposal["op"];t=proposal["target"];p=proposal["payload"]
  if op=="create":
   if not t or t in s.components:raise ValueError("create target")
   s.components[t]=Component(t,dict(p),True)
  elif op=="modify":
   if t not in s.components:raise ValueError("modify target")
   s.components[t].payload.update(p)
  elif op=="combine":
   src=p.get("sources",[])
   if not t or len(src)<2 or any(x not in s.components for x in src):raise ValueError("combine sources")
   merged={};[merged.update(s.components[x].payload) for x in src];merged.update(p.get("overrides",{}));s.components[t]=Component(t,merged,True)
  elif op=="freeze":
   if t not in s.components:raise ValueError("freeze target")
   s.components[t].active=False
  elif op=="prune":
   if t not in s.components:raise ValueError("prune target")
   s.archive[t]=deepcopy(s.components.pop(t))
  elif op=="restore":
   if t not in s.archive:raise ValueError("restore target")
   s.components[t]=s.archive.pop(t);s.components[t].active=True
  elif op=="noop":pass
  s.version+=1;return s
 def decide(self,state,proposal,baseline_cap:Capability,candidate_cap:Capability):
  shadow=self.apply_shadow(state,proposal)
  name=f'v{shadow.version}:{proposal["op"]}:{proposal["target"] or "-"}'
  accepted=self.frontier.consider(name,candidate_cap,baseline=baseline_cap)
  self.history.append({"proposal":proposal,"accepted":accepted,"from":state.version,"to":shadow.version if accepted else state.version})
  return (shadow if accepted else state),accepted

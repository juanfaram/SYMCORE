#!/usr/bin/env python3
"""Capability lifecycle policy: active -> hibernated -> archived; evidence is never deleted."""
class Lifecycle:
 def __init__(self,hibernate_after=5000,archive_after=20000):
  self.hibernate_after=hibernate_after;self.archive_after=archive_after;self.last_used={};self.state={}
 def touch(self,name,step):self.last_used[name]=step;self.state[name]="active"
 def update(self,step):
  for n,last in self.last_used.items():
   idle=step-last
   self.state[n]="archived" if idle>=self.archive_after else ("hibernated" if idle>=self.hibernate_after else "active")
  return dict(self.state)

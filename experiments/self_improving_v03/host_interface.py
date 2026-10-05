#!/usr/bin/env python3
"""Minimal host↔SYMCORE contract. The engine depends on behavior, not host internals."""
from __future__ import annotations
from typing import Any,Protocol,runtime_checkable
@runtime_checkable
class Host(Protocol):
    def predict(self,x:Any)->Any: ...
    def feedback(self,x:Any,y:Any)->None: ...
    def loss(self,prediction:Any,y:Any)->float: ...
    def capabilities(self)->dict: ...
@runtime_checkable
class AdaptiveHost(Host,Protocol):
    def candidates(self)->dict[str,Host]: ...
class PairedEvaluator:
    def step(self,control:Host,assisted:Host,x,y):
        pc=control.predict(x);ps=assisted.predict(x)
        lc=control.loss(pc,y);ls=assisted.loss(ps,y)
        control.feedback(x,y);assisted.feedback(x,y)
        return lc,ls

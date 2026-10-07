#!/usr/bin/env python3
"""Causal hysteretic intervention controller. Decisions use only prior control losses."""
import statistics
class InterventionController:
    def __init__(self,enter_z=1.25,exit_z=.50,min_history=128,window=256,exit_patience=24):
        self.enter_z=enter_z;self.exit_z=exit_z;self.min_history=min_history;self.window=window
        self.exit_patience=exit_patience;self.history=[];self.active=False;self.calm=0
    def decide(self):
        # Called before the current outcome exists.
        if len(self.history)<self.min_history:return self.active
        xs=self.history[-self.window:];mu=statistics.fmean(xs);sd=statistics.stdev(xs) if len(xs)>1 else 0.
        recent=statistics.fmean(xs[-24:])
        z=(recent-mu)/max(sd,1e-9)
        if not self.active and z>=self.enter_z:self.active=True;self.calm=0
        elif self.active:
            self.calm=self.calm+1 if z<=self.exit_z else 0
            if self.calm>=self.exit_patience:self.active=False;self.calm=0
        return self.active
    def observe(self,control_loss):self.history.append(float(control_loss))

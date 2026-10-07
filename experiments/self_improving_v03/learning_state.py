#!/usr/bin/env python3
"""Prospective learning-state z_H^learn computed strictly from information available at E."""
from __future__ import annotations
import math,statistics
from collections import deque
class LearningState:
 def __init__(self,window=256):
  self.window=window;self.loss=deque(maxlen=window);self.improvement=deque(maxlen=window)
 def observe(self,loss):
  x=float(loss)
  if self.loss:self.improvement.append(self.loss[-1]-x)
  self.loss.append(x)
 def vector(self):
  xs=list(self.loss);im=list(self.improvement)
  if len(xs)<16:return (0.,0.,1.)
  # 1) learning_velocity: robust local loss decrease per interaction (positive = improving).
  half=len(xs)//2
  velocity=(statistics.fmean(xs[:half])-statistics.fmean(xs[-half:]))/max(1,len(xs))
  # 2) saturation: recent improvement relative to its historical magnitude; 1 = little marginal improvement remains.
  recent=statistics.fmean(im[-min(64,len(im)):]) if im else 0.
  scale=statistics.fmean(abs(x) for x in im) if im else 0.
  saturation=1.-min(1.,max(0.,recent/max(1e-9,scale)))
  # 3) noise: robust-ish coefficient of local variation, bounded to avoid scale explosion.
  med=statistics.median(xs);mad=statistics.median(abs(x-med) for x in xs)
  noise=min(10.,mad/max(1e-9,abs(med)))
  return (velocity,saturation,noise)

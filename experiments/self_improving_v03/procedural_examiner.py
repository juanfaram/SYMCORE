#!/usr/bin/env python3
"""v1.2 procedural examiner whose regimes depend on learner behaviour."""
import random
from collections import deque,defaultdict
class ProceduralExaminer:
    def __init__(self,seed=707):
        self.rng=random.Random(seed);self.hist=defaultdict(lambda:deque(maxlen=200));self.step=0
    def observe(self,context,correct):
        self.hist[(context["domain"],context["difficulty"])].append(bool(correct));self.step+=1
    def generate(self):
        domains=("code","analysis","writing","forecast","rare");diffs=("simple","normal","hard")
        cells=[(d,q) for d in domains for q in diffs]
        def weakness(cell):
            xs=self.hist[cell];acc=sum(xs)/len(xs) if xs else .5
            novelty=1/(len(xs)+1);return (1-acc)+.4*novelty+self.rng.random()*.03
        d,q=max(cells,key=weakness)
        # when learner masters a cell, perturb the causal rule rather than repeating it forever
        mastered=len(self.hist[(d,q)])>=100 and sum(self.hist[(d,q)])/len(self.hist[(d,q)])>.88
        return {"task":"solve","domain":d,"difficulty":q,"perturbation":"invert_secondary" if mastered else "base"}

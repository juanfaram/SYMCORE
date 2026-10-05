#!/usr/bin/env python3
"""Diagnose cross-paradigm failure: compare routing overhead vs representation mismatch."""
import csv,json,math,statistics
from pathlib import Path
from main import ensure
def run():
 p=Path("data/pima.csv")
 if not p.exists():
  import urllib.request;urllib.request.urlretrieve("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",p)
 rows=[list(map(float,r)) for r in csv.reader(p.open())];sc=[20,200,130,100,900,70,1,100]
 # measure simple separability signal per feature; if no expert diversity exists, routing cannot add capability
 y=[int(r[-1]) for r in rows];scores={}
 for j in range(8):
  pos=[r[j]/sc[j] for r in rows if int(r[-1])==1];neg=[r[j]/sc[j] for r in rows if int(r[-1])==0]
  scores[str(j)]=abs(statistics.fmean(pos)-statistics.fmean(neg))/max(1e-9,statistics.pstdev([r[j]/sc[j] for r in rows]))
 out={"feature_effects":scores,"max_effect":max(scores.values()),"diagnosis":"router_needs_nonredundant_capabilities" if max(scores.values())<1 else "representation_signal_exists"}
 Path("artifacts/tabular_failure_diagnosis.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

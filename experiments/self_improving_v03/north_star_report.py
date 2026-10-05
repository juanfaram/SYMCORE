#!/usr/bin/env python3
"""Invariant north-star scorecard. Missing evidence is UNKNOWN."""
import json
from pathlib import Path
def maybe(*names):
 for n in names:
  p=Path("artifacts")/n
  if p.exists():return json.loads(p.read_text())
 return None
def run():
 host=maybe("real_host_control_report.json","real_host_control.json")
 cross=maybe("cross_host_report.json");lp=maybe("L_permutation_report.json");g=maybe("G_inflation_report.json")
 delta=None
 if host:
  if "vs_online_control" in host:delta=host["vs_online_control"]["gain"]
  else:delta=host.get("paired_gain")
 portable=cross.get("universal_pass") if cross else None
 A=lp.get("median_L") if lp else None
 out={"schema":"symcore.north-star.v2","delta_omega_host":delta,"portable_host_advantage":portable,
      "A_general":A,"dA_dE":None,"risk_bound_pass":None,
      "G_strict_count":g.get("strict_G_count") if g else None,
      "claims":{"delta_omega":(delta is not None and delta>0 and portable is True),
                "A":(A>0) if A is not None else None,"dA_dE":None,"risk":None}}
 Path("artifacts/north_star_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

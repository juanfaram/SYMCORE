#!/usr/bin/env python3
"""North-star scorecard: keep host capability growth separate from learning acceleration."""
import json
from pathlib import Path
def run():
 host=json.loads(Path("artifacts/real_host_control.json").read_text())
 cross=json.loads(Path("artifacts/cross_host_report.json").read_text())
 lp=json.loads(Path("artifacts/L_permutation_report.json").read_text())
 g=json.loads(Path("artifacts/G_inflation_report.json").read_text())
 out={"delta_omega_host":host.get("paired_gain",0),"portable_host_advantage":cross["universal_pass"],
      "A_general":lp["median_L"],"dA_dE_confirmed":False,
      "G_strict_count":g["strict_G_count"],
      "claims":{"host_growth":host.get("paired_gain",0)>0 and cross["universal_pass"],
                "general_learning_acceleration":lp["median_L"]>0,
                "accelerating_acceleration":False}}
 Path("artifacts/north_star_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

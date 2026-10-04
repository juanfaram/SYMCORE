#!/usr/bin/env python3
"""No-regression gate: progress must beat references AND retain learned skills."""
import json,sys
from pathlib import Path
def load(n):return json.loads((Path("artifacts")/n).read_text())
def main():
    failures=[]
    b=load("baseline_report.json");s=load("specialist_report.json");m=load("mixture_report.json")
    st=load("stress_report.json");cu=load("curriculum_report.json");fr=load("frontier_report.json")
    if not fr.get("passed") or not fr.get("accepted_frontier"):
        failures.append("capability frontier tournament produced no validated survivor")
    if len(fr.get("seeds",[]))<5:
        failures.append("capability frontier evidence requires at least 5 seeds")
    best_ref=min(b["persistence_mae"],b["same_hour_mae"],*s.values())
    if m["mixture_mae_last512"]>best_ref*1.05:
        failures.append(f"mixture MAE {m['mixture_mae_last512']} is >5% worse than best causal reference {best_ref}")
    if st["post_adaptation_accuracy"]<.80:
        failures.append(f"post-shift adaptation too low: {st['post_adaptation_accuracy']}")
    for phase in cu["phases"]:
        if phase["current_accuracy"]<.75:failures.append(f"weak acquisition {phase['phase']}: {phase['current_accuracy']}")
        for old in phase["retention"]:
            if old["accuracy"]<.45:failures.append(f"catastrophic forgetting {old['skill']} after {phase['phase']}: {old['accuracy']}")
    report={"passed":not failures,"best_reference_mae":best_ref,"failures":failures}
    Path("artifacts/gate_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 1 if failures else 0
if __name__=="__main__":sys.exit(main())

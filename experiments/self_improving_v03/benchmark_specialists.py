#!/usr/bin/env python3
import csv,json
from pathlib import Path
from seasonal import SeasonalMemory
def evaluate(path):
    models={"hour":SeasonalMemory(("hr",),8),"hour_workday":SeasonalMemory(("hr","workingday"),8),
            "hour_weekday":SeasonalMemory(("hr","weekday"),6),"hour_weather":SeasonalMemory(("hr","weathersit"),6)}
    errors={k:[] for k in models}
    with path.open() as f:
      for r in csv.DictReader(f):
        y=float(r["cnt"])
        for name,m in models.items():
            p=m.predict(r)
            if m.global_mem:errors[name].append(abs(p-y))
            m.learn(r,y)
    return {k:round(sum(v[-512:])/len(v[-512:]),3) for k,v in errors.items()}
if __name__=="__main__":
    p=Path("data/hour.csv")
    if not p.exists():
        from main import ensure;ensure(p)
    result=evaluate(p);Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/specialist_report.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

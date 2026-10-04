import csv,json
from pathlib import Path
from seasonal import SeasonalMemory
from mixture import AdaptiveMixture
def run():
 e={"hour":SeasonalMemory(("hr",),8),"hour_work":SeasonalMemory(("hr","workingday"),8),"hour_day":SeasonalMemory(("hr","weekday"),6),"hour_weather":SeasonalMemory(("hr","weathersit"),6)}
 m=AdaptiveMixture(e,.04);errs=[];snaps=[]
 with Path("data/hour.csv").open() as f:
  for i,r in enumerate(csv.DictReader(f),1):
   y=float(r["cnt"]);ctx="work" if r["workingday"]=="1" else "off";p,_,w=m.learn(r,y,ctx);errs.append(abs(p-y))
   if i%2000==0:snaps.append({"at":i,"context":ctx,"weights":{k:round(v,3) for k,v in w.items()}})
 out={"mixture_mae_last512":round(sum(errs[-512:])/512,3),"snapshots":snaps};Path("artifacts/mixture_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__":
 if not Path("data/hour.csv").exists():
  from main import ensure;ensure(Path("data/hour.csv"))
 run()

#!/usr/bin/env python3
"""Independent temporal stream replication on UCI Metro Interstate Traffic Volume."""
import csv,gzip,io,json,urllib.request
from pathlib import Path
from seasonal import SeasonalMemory
from intervention_controller import InterventionController
from risk_contract import risk_report,passes
URL="https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
def ensure_data(p):
 if p.exists():return
 import zipfile
 raw=urllib.request.urlopen(URL,timeout=60).read();z=zipfile.ZipFile(io.BytesIO(raw));name=[n for n in z.namelist() if n.endswith(".csv.gz")][0]
 p.parent.mkdir(exist_ok=True);p.write_bytes(gzip.decompress(z.read(name)))
def run():
 p=Path("data/metro_traffic.csv");ensure_data(p);rows=list(csv.DictReader(p.open()));base=SeasonalMemory(("hour","weekday"),8);rich=SeasonalMemory(("hour","weekday","weather_main"),6);ctl=InterventionController();control=[];sel=[];g=[];delta=[]
 for r in rows:
  dt=r["date_time"];date,time=dt.split();import datetime as D;x=D.datetime.fromisoformat(dt);r["hour"]=str(x.hour);r["weekday"]=str(x.weekday());y=float(r["traffic_volume"])
  bp=base.predict(r);rp=rich.predict(r);active=ctl.decide();be=abs(bp-y);re=abs(rp-y);control.append(be);sel.append(re if active else be);delta.append(be-re);g.append(int(active));ctl.observe(be);base.learn(r,y);rich.learn(r,y)
 split=int(len(rows)*.60);risk=risk_report(control[split:],sel[split:],.05);gain=sum(a-b for a,b in zip(control[split:],sel[split:]))/(len(rows)-split)
 rate=sum(g[split:])/(len(rows)-split);total=sum(delta[split:]);capt=sum(delta[i]*g[i] for i in range(split,len(rows)));R=capt/(rate*total) if rate and total>0 else None
 out={"dataset":"UCI Metro Interstate Traffic Volume","rows":len(rows),"split":.60,"controller_frozen":True,"active_fraction":rate,
      "mean_gain":gain,"R":R,"risk":risk,"risk_pass":passes(risk,.05)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/second_stream_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()

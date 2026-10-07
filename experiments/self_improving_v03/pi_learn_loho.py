#!/usr/bin/env python3
"""Leave-one-host-out falsification for Pi_learn using fixed opened-host V surfaces."""
from pi_learn import PiLearn,features
# Values are frozen outputs of run 37605406344; no re-opening or tuning against a new host.
SURFACES={
"bike":{1024:{256:-.7158203125,512:.217529296875,1024:4.6688232421875,2048:24.1368408203125},2048:{256:-1.4838169642857153,512:2.513706752232146,1024:24.873779296875,2048:49.7386474609375},4096:{256:-.0849609375,512:.183837890625,1024:.5615234375,2048:-.644775390625},8192:{256:-.45166015625,512:1.877685546875,1024:-1.156005859375,2048:3.3597412109375}},
"metro":{1024:{256:11.844935825892861,512:12.02958519345242,1024:14.559863281249989,2048:52.669052850632454},2048:{256:13.1787109375,512:25.0146484375,1024:-13.3472900390625,2048:-28.498046875},4096:{256:20.92626953125,512:32.546875,1024:42.6112060546875,2048:60.88665771484375},8192:{256:1.484375,512:13.352783203125,1024:11.2353515625,2048:21.0391845703125},16384:{256:14.74951171875,512:22.609375,1024:23.785888671875,2048:7.66351318359375}},
"beijing":{1024:{256:14.53614676339285,512:4.31285342261905,1024:2.48008626302083,2048:1.914522879464279},2048:{256:3.738295200892857,512:2.98802083333333,1024:8.605995396205358,2048:6.106278483072913},4096:{256:10.337248883928574,512:1.836421130952381,1024:1.8317347935267847,2048:.3162580217633959},8192:{256:6.29736328125,512:7.685302734375,1024:5.4139404296875,2048:-5.5655517578125},16384:{256:-5.13525390625,512:1.199951171875,1024:3.700439453125,2048:5.90692138671875}}}
def oracle(v,margin=.5):
 if v>margin:return "LEARN"
 if v<-margin:return "FREEZE"
 return "MEASURE"
def run():
 out={}
 for held in SURFACES:
  p=PiLearn(k=12,margin=.5)
  for host,s in SURFACES.items():
   if host==held:continue
   for E,hs in s.items():
    for h,v in hs.items():p.observe(features(E,h),v)
  rows=[]
  for E,hs in SURFACES[held].items():
   for h,v in hs.items():
    pred,_=p.decide(features(E,h));truth=oracle(v);rows.append((pred,truth,v))
  decisive=[x for x in rows if x[1]!="MEASURE"];correct=sum(a==b for a,b,_ in decisive);coverage=sum(a!="MEASURE" for a,_,_ in rows)/len(rows)
  out[held]={"decisive_accuracy":correct/max(1,len(decisive)),"decision_coverage":coverage,"n":len(rows),"errors":sum(a!=b for a,b,_ in decisive)}
 print(__import__("json").dumps(out,indent=2));return out
if __name__=="__main__":run()

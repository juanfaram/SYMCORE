"""Reproducible differential + multiscale benchmark for Ω corpus v1."""
import json, statistics, time, tracemalloc, platform, sys
from benchmarks import corpus, candidates

CASES = {
    "reduction_sum": lambda n: ([i % 17 - 8 for i in range(n)],),
    "map_filter_fusion": lambda n: ([i % 31 - 15 for i in range(n)],),
    "repeated_membership": lambda n: ([i for i in range(n)], [i % (n + 1) for i in range(max(1,n//2))]),
    "prefix_sum": lambda n: ([i % 17 - 8 for i in range(n)],),
    "repeated_subproblem": lambda n: (n,),
}
SCALES = {
    "reduction_sum":[100,1000,10000,100000],
    "map_filter_fusion":[100,1000,10000,100000],
    "repeated_membership":[100,1000,5000,10000],
    "prefix_sum":[100,1000,10000,100000],
    "repeated_subproblem":[10,100,1000,5000],
}

def outcome(fn,args):
    try: return ("RET",fn(*args))
    except Exception as e: return ("EXC",type(e).__qualname__,str(e))

def measure(fn,args,reps=21):
    for _ in range(5): outcome(fn,args)
    ts=[]; peaks=[]
    for _ in range(reps):
        tracemalloc.start(); t=time.perf_counter_ns(); outcome(fn,args)
        ts.append(time.perf_counter_ns()-t); _,p=tracemalloc.get_traced_memory(); tracemalloc.stop(); peaks.append(p)
    med=int(statistics.median(ts))
    mad=int(statistics.median(abs(x-med) for x in ts))
    return {"median_ns":med,"MAD_ns":mad,"peak_bytes":int(statistics.median(peaks)),"repetitions":reps}

def main():
    rows=[]; failures=[]
    for name,make_args in CASES.items():
        base=getattr(corpus,name); cand=getattr(candidates,name)
        for n in SCALES[name]:
            args=make_args(n)
            a,b=outcome(base,args),outcome(cand,args)
            if a!=b:
                failures.append({"problem":name,"N":n,"baseline":repr(a),"candidate":repr(b)})
                continue
            bm=measure(base,args); cm=measure(cand,args)
            rows.append({"problem":name,"N":n,"baseline":bm,"candidate":cm,
                         "speedup":bm["median_ns"]/cm["median_ns"] if cm["median_ns"] else None})
    out={"epistemic":"[M] local run; no universal performance claim",
         "environment":{"python":sys.version,"platform":platform.platform()},
         "failures":failures,"measurements":rows}
    print(json.dumps(out,indent=2))
    return 1 if failures else 0

if __name__=="__main__":
    raise SystemExit(main())

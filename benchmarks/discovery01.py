import json,statistics,time
from omega.discovery01.datasets import make_case,VALIDATION
from omega.discovery01.oracles import recompute_reference,oracle_optimized_recompute
from omega.discovery01.realization import discovered_realization
from omega.discovery01.search import synthesize

def med(fn,n=11):
    for _ in range(2): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    m=statistics.median(xs); mad=statistics.median([abs(x-m) for x in xs])
    return m,mad,xs

rows=[]
for seed,n,steps in VALIDATION:
    s,ops=make_case(seed,n,steps)
    funcs={"reference":lambda:recompute_reference(s,ops),
           "hostile":lambda:oracle_optimized_recompute(s,ops),
           "discovered":lambda:discovered_realization(s,ops)}
    row={"seed":seed,"N":n,"steps":steps}
    for name,fn in funcs.items():
        m,mad,xs=med(fn); row[name+"_p50_ns"]=m; row[name+"_mad_ns"]=mad; row[name+"_samples"]=xs
    row["speedup_vs_hostile"]=row["hostile_p50_ns"]/row["discovered_p50_ns"]
    row["speedup_vs_reference"]=row["reference_p50_ns"]/row["discovered_p50_ns"]
    rows.append(row)
print(json.dumps({
 "search":synthesize(),
 "complexity":{"reference":"O(sum_t |S_t|) ~= O(T*N)","discovered":"O(N+T)","observable_update":"O(1) per operation","lower_bound":"Omega(N+T) to consume initial state and operation stream"},
 "rows":rows},sort_keys=True))

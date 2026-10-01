import json, random, statistics, time
from omega.reformulation.membership import linear_membership, omega_membership, hostile_membership

def med(fn,n=9):
    for _ in range(2): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs),xs

rng=random.Random(3303)
rows=[]
for N,Q in [(100,100),(1000,1000),(5000,5000),(10000,10000)]:
    values=[rng.randrange(0,N*4+1) for _ in range(N)]
    queries=[rng.randrange(0,N*4+1) for _ in range(Q)]
    funcs={"linear":lambda:linear_membership(values,queries),
           "omega":lambda:omega_membership(values,queries),
           "hostile":lambda:hostile_membership(values,queries)}
    row={"N":N,"Q":Q}
    for name,fn in funcs.items():
        t,s=med(fn)
        row[name+"_ns"]=t; row[name+"_samples"]=s
    row["omega_vs_linear"]=row["linear_ns"]/row["omega_ns"]
    row["hostile_vs_linear"]=row["linear_ns"]/row["hostile_ns"]
    row["omega_vs_hostile"]=row["hostile_ns"]/row["omega_ns"]
    rows.append(row)
print(json.dumps({"complexity":{"linear":"O(N*Q)","omega":"expected O(N+Q)","hostile":"expected O(N+Q)","lower_bound":"Omega(N+Q) to consume both inputs under this batch contract"},"rows":rows},sort_keys=True))

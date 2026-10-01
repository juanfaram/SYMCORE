import json, os, random, time
from omega.open_discovery01.incumbent import INCUMBENT_71
from omega.open_discovery01.verifier import evaluate

seed=int(os.environ.get("OMEGA_SEED","17001"))
budget=float(os.environ.get("OMEGA_SECONDS","2400"))
rng=random.Random(seed)
deadline=time.time()+budget

pairs=[(a,b) for a in range(17) for b in range(a+1,17)]

# Start from every one-comparator deletion of the frozen incumbent.
pool=[]
for i in range(len(INCUMBENT_71)):
    n=INCUMBENT_71[:i]+INCUMBENT_71[i+1:]
    pool.append((evaluate(n),n))
pool.sort(key=lambda x:x[0])
best_bad,best=pool[0]
current_bad,current=best_bad,best[:]
accepted=0; evaluated=len(pool)

def mutate(net):
    z=net[:]
    mode=rng.randrange(4)
    if mode==0:
        i=rng.randrange(70); z[i]=rng.choice(pairs)
    elif mode==1:
        i=rng.randrange(70); a,b=z[i]
        if rng.random()<.5: a=rng.randrange(0,b)
        else: b=rng.randrange(a+1,17)
        z[i]=(a,b)
    elif mode==2:
        i,j=rng.sample(range(70),2); z[i],z[j]=z[j],z[i]
    else:
        i=rng.randrange(69); z[i],z[i+1]=z[i+1],z[i]
    return z

temp=max(1.0,best_bad/20)
last_report=time.time()
while time.time()<deadline and best_bad:
    cand=mutate(current)
    bad=evaluate(cand); evaluated+=1
    delta=bad-current_bad
    # simulated annealing with periodic restart from elite deletion candidates
    accept=bad<=current_bad or rng.random()<pow(2.718281828,-max(0,delta)/max(1,temp))
    if accept:
        current_bad,current=bad,cand; accepted+=1
    if bad<best_bad:
        best_bad,best=bad,cand
        print(json.dumps({"event":"improvement","seed":seed,"bad":best_bad,"evaluated":evaluated,"network":best}),flush=True)
    temp*=0.9995
    if temp<0.05 or rng.random()<0.0005:
        temp=max(1.0,best_bad/20)
        if rng.random()<.5:
            current_bad,current=best_bad,best[:]
        else:
            current_bad,current=rng.choice(pool[:min(12,len(pool))]); current=current[:]

print(json.dumps({"event":"final","seed":seed,"bad":best_bad,"evaluated":evaluated,"accepted":accepted,"network":best}),flush=True)
if best_bad==0:
    open(f"FOUND_70_{seed}.json","w").write(json.dumps(best))

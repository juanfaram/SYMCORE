import cProfile, io, json, pstats, time, torch
from symcore import compress
torch.manual_seed(404)
B,L,D=2,256,64
base=torch.randn(B,L//2,D)
X=torch.cat([base,torch.flip(base,dims=[1])],dim=1)

pr=cProfile.Profile(); pr.enable()
t0=time.perf_counter_ns()
for _ in range(5): compress(X,window_size=16,epsilon=0.0)
elapsed=time.perf_counter_ns()-t0
pr.disable()
s=io.StringIO()
pstats.Stats(pr,stream=s).sort_stats("cumtime").print_stats(25)
print(json.dumps({"elapsed_ns":elapsed,"iterations":5,"mean_ns":elapsed/5}))
print(s.getvalue())

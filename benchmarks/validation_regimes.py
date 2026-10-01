import json, statistics, time, torch
from symcore import compress, decompress
D=64
class M(torch.nn.Module):
    def __init__(self): super().__init__(); self.a=torch.nn.MultiheadAttention(D,4,batch_first=True)
    def forward(self,x): return self.a(x,x,x)[0]
def med(fn,n=12):
    for _ in range(3): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs)
rows=[]
for seed in [1101,1102,1103,1104]:
  torch.manual_seed(seed)
  for L in [1536,2048]:
    for p in [2,4,8]:
      model=M().eval()
      basepat=torch.randn(1,p,D); X=basepat.repeat(1,L//p,1)
      with torch.no_grad(): xc,pos=compress(X,window_size=16,epsilon=1e-6)
      def base():
        with torch.no_grad(): model(X)
      def opt():
        with torch.no_grad():
          y,m=compress(X,window_size=16,epsilon=1e-6); model(y); decompress(y,m,L)
      b=med(base); o=med(opt)
      rows.append({"seed":seed,"L":L,"period":p,"r":X.shape[1]/xc.shape[1],"baseline_ns":b,"optimized_ns":o,"speedup":b/o})
print(json.dumps(rows))

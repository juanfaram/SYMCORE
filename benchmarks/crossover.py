import json, statistics, time, torch
from symcore import compress, decompress
torch.manual_seed(888)
D=64
class M(torch.nn.Module):
    def __init__(self): super().__init__(); self.a=torch.nn.MultiheadAttention(D,4,batch_first=True)
    def forward(self,x): return self.a(x,x,x)[0]
model=M().eval()
def med(fn,n=8):
    for _ in range(2): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs)
rows=[]
for L in [128,256,512,1024,1536]:
    B=1
    pb=torch.randn(B,4,D); X=pb.repeat(1,L//4,1)
    with torch.no_grad(): xc,pos=compress(X,window_size=16,epsilon=1e-6)
    def base():
        with torch.no_grad(): model(X)
    def opt():
        with torch.no_grad():
            y,p=compress(X,window_size=16,epsilon=1e-6); model(y); decompress(y,p,L)
    b=med(base); o=med(opt)
    rows.append({"L":L,"r":X.shape[1]/xc.shape[1],"baseline_ns":b,"optimized_ns":o,"speedup":b/o})
print(json.dumps(rows))

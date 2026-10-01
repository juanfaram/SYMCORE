import json, platform, statistics, time
import torch
from symcore import compress, decompress

torch.manual_seed(2026)
B,L,D = 4,512,64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = torch.randn(B,L//2,D, device=device)
X = torch.cat([half, torch.flip(half,dims=[1])],dim=1)

class MockTransformer(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.attn=torch.nn.MultiheadAttention(d,4,batch_first=True)
    def forward(self,x):
        return self.attn(x,x,x)[0]

model=MockTransformer(D).to(device).eval()

def sync():
    if device.type=="cuda": torch.cuda.synchronize()

def baseline():
    with torch.no_grad(): model(X)
    sync()

def optimized():
    with torch.no_grad():
        xc,pos=compress(X,window_size=16,epsilon=0.0)
        model(xc)
        # Reconstruction is included because this lab contract treats it as observable.
        decompress(xc,pos,L)
    sync()

for _ in range(10): baseline(); optimized()

def samples(fn,n=50):
    out=[]
    for _ in range(n):
        sync(); t=time.perf_counter_ns(); fn(); sync()
        out.append(time.perf_counter_ns()-t)
    return out

b=samples(baseline); o=samples(optimized)
result={
 "device":str(device),"python":platform.python_version(),"torch":torch.__version__,
 "baseline_ns":b,"optimized_ns":o,
 "baseline_p50_ns":statistics.median(b),"optimized_p50_ns":statistics.median(o),
 "speedup_p50":statistics.median(b)/statistics.median(o)
}
print(json.dumps(result))

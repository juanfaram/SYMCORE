import json, statistics, time, torch
from symcore import compress, decompress

torch.manual_seed(777)
B,L,D=2,256,64
device=torch.device("cpu")

class MockTransformer(torch.nn.Module):
    def __init__(self,d):
        super().__init__(); self.attn=torch.nn.MultiheadAttention(d,4,batch_first=True)
    def forward(self,x): return self.attn(x,x,x)[0]
model=MockTransformer(D).eval()

def make_regimes():
    random=torch.randn(B,L,D)
    gh=torch.randn(B,L//2,D); global_mirror=torch.cat([gh,torch.flip(gh,[1])],1)
    mh=torch.randn(B,8,D); mb=torch.cat([mh,torch.flip(mh,[1])],1); local_mirror=mb.repeat(1,L//16,1)
    pb=torch.randn(B,4,D); periodic=pb.repeat(1,L//4,1)
    sh=torch.randn(B,8,D); sb=torch.cat([sh,sh*2.0],1); scale=sb.repeat(1,L//16,1)
    return {"random":random,"global_mirror":global_mirror,"local_mirror":local_mirror,"periodic":periodic,"scale":scale}

def timed(fn,n=20):
    for _ in range(3): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs),xs

results={}
for name,X in make_regimes().items():
    with torch.no_grad():
        xc,pos=compress(X,window_size=16,epsilon=1e-6)
    ratio=X.shape[1]/xc.shape[1]
    def base():
        with torch.no_grad(): model(X)
    def comp_only():
        with torch.no_grad(): compress(X,window_size=16,epsilon=1e-6)
    def opt():
        with torch.no_grad():
            y,p=compress(X,window_size=16,epsilon=1e-6)
            model(y); decompress(y,p,L)
    b,_=timed(base); c,_=timed(comp_only); o,_=timed(opt)
    results[name]={"compression_ratio":ratio,"baseline_p50_ns":b,"compress_p50_ns":c,"optimized_p50_ns":o,"e2e_speedup":b/o}
print(json.dumps(results,sort_keys=True))

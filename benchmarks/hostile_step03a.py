import json, statistics, time, torch
from symcore import compress, decompress

torch.manual_seed(3301)
B,L,D=1,2048,64
pattern=torch.randn(B,2,D)
X=pattern.repeat(1,L//2,1)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=torch.nn.MultiheadAttention(D,4,batch_first=True)
    def forward(self,x):
        return self.attn(x,x,x)[0]

eager=Model().eval()

def med(fn,n=12):
    for _ in range(3): fn()
    xs=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs), xs

def eager_base():
    with torch.no_grad(): eager(X)

def omega_eager():
    with torch.no_grad():
        y,p=compress(X,window_size=16,epsilon=1e-6)
        eager(y)
        decompress(y,p,L)

out={}
out["eager_ns"],out["eager_samples"]=med(eager_base)
out["omega_eager_ns"],out["omega_eager_samples"]=med(omega_eager)
out["omega_eager_vs_eager"]=out["eager_ns"]/out["omega_eager_ns"]

try:
    compiled=torch.compile(eager)
    def compiled_base():
        with torch.no_grad(): compiled(X)
    def omega_compiled():
        with torch.no_grad():
            y,p=compress(X,window_size=16,epsilon=1e-6)
            compiled(y)
            decompress(y,p,L)
    out["compiled_ns"],out["compiled_samples"]=med(compiled_base)
    out["omega_compiled_ns"],out["omega_compiled_samples"]=med(omega_compiled)
    out["omega_eager_vs_compiled"]=out["compiled_ns"]/out["omega_eager_ns"]
    out["omega_compiled_vs_compiled"]=out["compiled_ns"]/out["omega_compiled_ns"]
    out["compile_status"]="OK"
except Exception as e:
    out["compile_status"]="FAILED"
    out["compile_error"]=type(e).__name__ + ": " + str(e)[:500]

with torch.no_grad():
    yc,pm=compress(X,window_size=16,epsilon=1e-6)
out["compression_ratio"]=X.shape[1]/yc.shape[1]
print(json.dumps(out,sort_keys=True))

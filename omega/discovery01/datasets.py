import random

def make_case(seed,n,steps):
    rng=random.Random(seed)
    state=[rng.randrange(-1000,1001) for _ in range(n)]
    live=list(state); ops=[]
    for _ in range(steps):
        if not live or rng.random()<0.62:
            x=rng.randrange(-1000,1001); ops.append(("ADD",x)); live.append(x)
        else:
            i=rng.randrange(len(live)); x=live.pop(i); ops.append(("REMOVE",x))
    return state,ops

DEV=[(4101,128,512),(4101,512,1024)]
VALIDATION=[(4201,1024,2048),(4202,4096,4096),(4203,8192,8192)]
BLIND=[(49001,16384,16384),(49003,32768,32768)]

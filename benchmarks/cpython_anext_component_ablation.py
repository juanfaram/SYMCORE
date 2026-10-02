import asyncio,json,statistics,time,platform,sys
ITEMS=500_000; REPEATS=9; SENTINEL=object()
async def items():
    for v in range(ITEMS): yield v
def w_simple(x): return x.__anext__()
def w_default(x, default=SENTINEL):
    aw=x.__anext__()
    if default is SENTINEL: return aw
    return aw
def w_lookup(x, default=SENTINEL):
    cls=type(x)
    try: m=cls.__anext__
    except AttributeError:
        raise TypeError(f"{cls.__name__!r} object is not an async iterator") from None
    aw=m(x)
    if default is SENTINEL: return aw
    return aw
async def run(mode):
    s=items(); t=time.perf_counter_ns()
    if mode=="builtin":
        for _ in range(ITEMS): await anext(s)
    elif mode=="simple":
        for _ in range(ITEMS): await w_simple(s)
    elif mode=="default":
        for _ in range(ITEMS): await w_default(s)
    elif mode=="lookup":
        for _ in range(ITEMS): await w_lookup(s)
    elif mode=="direct":
        for _ in range(ITEMS): await s.__anext__()
    return (time.perf_counter_ns()-t)/ITEMS
async def main():
    out={"python":sys.version,"platform":platform.platform()}
    for mode in ["builtin","lookup","default","simple","direct"]:
        xs=[await run(mode) for _ in range(REPEATS)]; m=statistics.median(xs)
        out[mode]={"median_ns_item":m,"mad_ns_item":statistics.median([abs(x-m) for x in xs]),"samples":xs}
    print(json.dumps(out,sort_keys=True))
asyncio.run(main())

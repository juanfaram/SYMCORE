import asyncio, json, statistics, time, platform, sys
ITEMS=500_000; REPEATS=9
async def items():
    for v in range(ITEMS): yield v
def py_wrapper(x):
    return x.__anext__()
async def run(mode):
    s=items(); t=time.perf_counter_ns()
    if mode=="builtin":
        for _ in range(ITEMS): await anext(s)
    elif mode=="wrapper":
        for _ in range(ITEMS): await py_wrapper(s)
    elif mode=="direct":
        for _ in range(ITEMS): await s.__anext__()
    return (time.perf_counter_ns()-t)/ITEMS
async def main():
    out={"python":sys.version,"platform":platform.platform()}
    for mode in ["builtin","wrapper","direct"]:
        xs=[await run(mode) for _ in range(REPEATS)]
        m=statistics.median(xs)
        out[mode]={"samples":xs,"median_ns_item":m,
                   "mad_ns_item":statistics.median([abs(x-m) for x in xs])}
    print(json.dumps(out,sort_keys=True))
asyncio.run(main())

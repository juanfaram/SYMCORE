import asyncio, json, statistics, time, platform, sys
ITEMS=500_000
REPEATS=9
async def items():
    for value in range(ITEMS):
        yield value
async def consume_with_anext():
    stream=items(); t=time.perf_counter_ns()
    for _ in range(ITEMS): await anext(stream)
    return (time.perf_counter_ns()-t)/ITEMS
async def consume_with_async_for():
    t=time.perf_counter_ns()
    async for _ in items(): pass
    return (time.perf_counter_ns()-t)/ITEMS
async def main():
    out={"python":sys.version,"platform":platform.platform(),"items":ITEMS,"repeats":REPEATS}
    for name,fn in [("anext",consume_with_anext),("async_for",consume_with_async_for)]:
        xs=[await fn() for _ in range(REPEATS)]
        out[name]={"samples_ns_item":xs,"median_ns_item":statistics.median(xs),
                   "mad_ns_item":statistics.median([abs(x-statistics.median(xs)) for x in xs])}
    print(json.dumps(out,sort_keys=True))
asyncio.run(main())

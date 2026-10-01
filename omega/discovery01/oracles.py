def recompute_reference(initial, operations):
    state=list(initial)
    out=[]
    for op,x in operations:
        if op=="ADD":
            state.append(x)
        elif op=="REMOVE":
            state.remove(x)
        else:
            raise ValueError(op)
        out.append(sum(v*v for v in state))
    return out

def oracle_optimized_recompute(initial, operations):
    # Stronger implementation of the same algorithmic strategy:
    # local bindings and explicit loop avoid generator overhead, but still rescans state.
    state=list(initial); out=[]; append=state.append; remove=state.remove
    for op,x in operations:
        if op=="ADD": append(x)
        else: remove(x)
        total=0
        for v in state: total += v*v
        out.append(total)
    return out

ORACLE_PROPOSALS=[
    "use local bindings",
    "replace generator sum with explicit loop",
    "batch output allocation"
]

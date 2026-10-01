def discovered_realization(initial, operations):
    state=list(initial)
    A=sum(v*v for v in state)
    out=[]
    for op,x in operations:
        if op=="ADD":
            state.append(x); A += x*x
        elif op=="REMOVE":
            state.remove(x); A -= x*x
        else:
            raise ValueError(op)
        out.append(A)
    return out

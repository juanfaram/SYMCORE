def discovered_realization(initial, operations):
    # Contract guarantees REMOVE(x) is valid and does not expose final collection state.
    # Therefore the materialized collection is not an observable and is unnecessary.
    A=sum(v*v for v in initial)
    out=[]
    for op,x in operations:
        if op=="ADD":
            A += x*x
        elif op=="REMOVE":
            A -= x*x
        else:
            raise ValueError(op)
        out.append(A)
    return out

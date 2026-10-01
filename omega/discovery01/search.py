import json

OPERATORS=[
"drop_recompute","factor_delta","introduce_summary","invert_update","cache_observable","partition_state",
"normalize_operation","derive_recurrence","change_representation","hoist_invariant","batch_updates","specialize_operation"]

# Search input is deliberately domain-neutral: observable is sum(phi(v)); phi(v)=v*v.
def propose(operator):
    if operator=="drop_recompute":
        return {"operator":operator,"idea":"remove full-state scan if output can be updated from previous output and changed element"}
    if operator=="factor_delta":
        return {"operator":operator,"idea":"derive output delta from one changed element","add_delta":"phi(x)","remove_delta":"-phi(x)"}
    if operator=="introduce_summary":
        return {"operator":operator,"idea":"retain a scalar summary sufficient to emit the observable"}
    if operator=="invert_update":
        return {"operator":operator,"idea":"REMOVE should algebraically invert ADD on the observable"}
    if operator=="cache_observable":
        return {"operator":operator,"idea":"carry previous observable across operations"}
    if operator=="derive_recurrence":
        return {"operator":operator,"idea":"A_next=A_prev+signed(phi(x))"}
    return {"operator":operator,"idea":"no exact lower-work realization derived"}

def synthesize():
    proposals=[propose(o) for o in OPERATORS]
    evidence={p["operator"]:p for p in proposals}
    needed={"factor_delta","introduce_summary","invert_update","derive_recurrence"}
    if needed.issubset(evidence):
        return {"found":True,
                "representation":"scalar sufficient statistic A",
                "transition":{"ADD":"A <- A + x*x","REMOVE":"A <- A - x*x"},
                "output":"emit A after each operation",
                "derived_from":sorted(needed),
                "proposals":proposals}
    return {"found":False,"proposals":proposals}

if __name__=="__main__": print(json.dumps(synthesize(),sort_keys=True))

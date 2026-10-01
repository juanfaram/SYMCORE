import json
traps=[
 {"id":"epsilon_zero_false_green","naive_accepts":True,"omega_accepts":False},
 {"id":"forward_only_speedup","naive_accepts":True,"omega_accepts":False},
 {"id":"modeled_energy_as_measured","naive_accepts":True,"omega_accepts":False},
]
print(json.dumps({"traps":traps,"naive_invalid_claims_accepted":sum(t["naive_accepts"] for t in traps),"omega_invalid_claims_accepted":sum(t["omega_accepts"] for t in traps)}))

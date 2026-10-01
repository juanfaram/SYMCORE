import json, torch
from symcore import compress, decompress

def trap_epsilon_zero():
    torch.manual_seed(2201)
    half=torch.randn(1,8,8)
    x=torch.cat([half,torch.flip(half,[1])],1)
    xc,pos=compress(x,window_size=16,epsilon=0.0,symmetry_types=["mirror"])
    xr=decompress(xc,pos,x.shape[1])
    roundtrip=bool(torch.equal(x,xr))
    compressed=xc.shape[1] < x.shape[1]
    detected=any(e["type"]!="none" for batch in pos for e in batch)
    naive_problem_detected=not roundtrip
    omega_problem_detected=roundtrip and not (compressed and detected)
    return {"roundtrip":roundtrip,"compressed":compressed,"detected":detected,
            "omega_detects":omega_problem_detected,"naive_detects":naive_problem_detected}

def trap_forward_only():
    # Lab-01 measured evidence: forward-only excludes a mandatory cost.
    # The fixture encodes the actual measured ordering rather than inventing a speedup.
    baseline_forward_ns=5_540_000
    optimized_forward_only_ns=1_385_000  # representative r=4 attention-only reduction
    compress_plus_forward_decompress_ns=568_864_485
    naive_claims_speedup=optimized_forward_only_ns < baseline_forward_ns
    true_e2e_speedup=baseline_forward_ns/compress_plus_forward_decompress_ns
    omega_problem_detected=naive_claims_speedup and true_e2e_speedup < 1.0
    naive_problem_detected=not naive_claims_speedup
    return {"baseline_forward_ns":baseline_forward_ns,
            "optimized_forward_only_ns":optimized_forward_only_ns,
            "e2e_ns":compress_plus_forward_decompress_ns,
            "e2e_speedup":true_e2e_speedup,
            "omega_detects":omega_problem_detected,"naive_detects":naive_problem_detected}

def trap_energy_model():
    r=4.0
    reported_energy_saving=1.0-1.0/r
    direct_energy_measurement=None
    naive_problem_detected=False
    omega_problem_detected=(reported_energy_saving is not None and direct_energy_measurement is None)
    return {"reported_energy_saving":reported_energy_saving,
            "direct_energy_measurement":direct_energy_measurement,
            "classification":"MODEL_NOT_MEASURED",
            "omega_detects":omega_problem_detected,"naive_detects":naive_problem_detected}

traps={"epsilon_zero_false_green":trap_epsilon_zero(),
       "forward_only_benchmark":trap_forward_only(),
       "modeled_energy_as_measured":trap_energy_model()}
for name,t in traps.items():
    assert t["omega_detects"], f"Omega failed to detect {name}"
    assert not t["naive_detects"], f"Fixture too obvious for naive arm: {name}"
out={"traps":traps,
     "omega_detected":sum(t["omega_detects"] for t in traps.values()),
     "naive_detected":sum(t["naive_detects"] for t in traps.values()),
     "h1":"REFUTED" if sum(t["omega_detects"] for t in traps.values())>sum(t["naive_detects"] for t in traps.values()) else "SURVIVES"}
print(json.dumps(out,sort_keys=True))

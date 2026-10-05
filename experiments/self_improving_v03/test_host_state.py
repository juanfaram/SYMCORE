from host_state import HostState
def test_host_state_is_online_and_finite():
 s=HostState()
 for i in range(20):s.observe(i/10,(i-1)/10,[i,i+1])
 v=s.vector();assert set(v)=={"error_level","error_velocity","recent_regret","expert_disagreement","change_probability"}
 assert 0<=v["change_probability"]<=1

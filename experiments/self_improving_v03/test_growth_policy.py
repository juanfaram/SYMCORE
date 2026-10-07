from growth_policy import GrowthPolicy
def test_growth_policy_uses_only_state_and_external_feedback():
 p=GrowthPolicy(5);s={"error_velocity":1,"change_probability":.8,"recent_regret":1,"expert_disagreement":1}
 for _ in range(8):p.observe(s,1,False)
 assert p.decide(s)

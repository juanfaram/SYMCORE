from growth_policy import GrowthPolicy
def test_pi1_learns_growth_region():
 p=GrowthPolicy(5);s={"error_velocity":1,"change_probability":.8,"recent_regret":1,"expert_disagreement":1}
 for _ in range(10):p.observe(s,2,False)
 assert p.decide(s)

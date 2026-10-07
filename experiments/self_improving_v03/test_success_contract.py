from capability_factory import CapabilityFactory
def test_factory_carries_success_contract():
 gap={"context":{"task":"solve","domain":"novel","difficulty":"hard"},"best":.3}
 s=CapabilityFactory().propose(gap,[{"name":"parent","context":{"difficulty":"hard"}}],success_action="deep")
 assert s.success_action=="deep" and s.success_threshold==.80

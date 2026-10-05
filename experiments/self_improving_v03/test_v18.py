from host_advantage import paired_advantage
def test_paired_advantage_requires_confident_gain():
 assert paired_advantage([10]*100,[5]*100)["passed"]
 assert not paired_advantage([10]*100,[10]*100)["passed"]

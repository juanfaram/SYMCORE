from pi_learn import PiLearn,features
def test_measure_is_real_action_under_uncertainty():
 p=PiLearn();p.observe(features(1000,256),0.0);assert p.decide(features(1000,256))[0]=="MEASURE"
def test_clear_positive_and_negative_can_separate():
 p=PiLearn(k=2,margin=.5)
 for e in (1000,1100):p.observe(features(e,256),10)
 assert p.decide(features(1050,256))[0]=="LEARN"
 q=PiLearn(k=2,margin=.5)
 for e in (1000,1100):q.observe(features(e,256),-10)
 assert q.decide(features(1050,256))[0]=="FREEZE"

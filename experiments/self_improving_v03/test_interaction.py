from interaction_learner import InteractionLearner
def test_feedback_changes_policy(tmp_path):
    l=InteractionLearner(["a","b"],epsilon=0,log=tmp_path/"x.jsonl")
    c={"task":"x","difficulty":"hard","domain":"test"}
    for _ in range(200):l.feedback("b",c,1);l.feedback("a",c,-1)
    assert l.choose(c)=="b"
def test_reward_is_clamped(tmp_path):
    l=InteractionLearner(["a"],log=tmp_path/"x")
    l.feedback("a",{},99)
    assert l.arms["a"].reward==1

from frontier_tournament import run, evaluate_policy

def test_frontier_tournament_is_reproducible(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    a=run((11,23,37,51,73))
    b=run((11,23,37,51,73))
    assert a==b
    assert a["passed"]
    assert a["accepted_frontier"]
    assert all(len(v["trials"])==5 for v in a["candidates"].values())

def test_policies_expose_real_tradeoffs():
    stable=evaluate_policy("stable",11,2000)
    plastic=evaluate_policy("plastic",11,2000)
    assert stable.retention > plastic.retention
    assert stable.efficiency > plastic.efficiency

from research_scheduler import ResearchScheduler
def test_scheduler_moves_budget_to_productive_line():
 s=ResearchScheduler(["a","b"],1)
 s.record("a",.8,cost=1,passed=True);s.record("b",-1,cost=2,passed=False)
 for _ in range(20):
  x=s.choose();s.record(x,.8 if x=="a" else -1,cost=1 if x=="a" else 2,passed=x=="a")
 assert s.lines["a"].n>s.lines["b"].n

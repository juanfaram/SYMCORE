from context_router import ContextRouter
def test_router_learns_contextual_winner():
 r=ContextRouter(2);x=[.1,.2,.3]
 for _ in range(100):r.feedback(x,[.8,.1])
 assert r.choose(x)==1

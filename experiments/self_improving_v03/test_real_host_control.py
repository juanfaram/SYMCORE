from real_host_control import Host,SymcoreHost,stats
ROW={"hr":"8","weekday":"1","workingday":"1","weathersit":"1"}
def test_all_three_arms_learn_causally():
 b=Host(("hr",));o=Host(("hr","workingday"));s=SymcoreHost()
 first=[b.step(ROW,100),o.step(ROW,100),s.step(ROW,100)]
 second=[b.step(ROW,100),o.step(ROW,100),s.step(ROW,100)]
 assert first==[100,100,100]
 assert all(y<=x for x,y in zip(first,second))
def test_A_is_zero_for_identical_losses():
 assert stats([1,2,3],[1,2,3])["A"]==0

from real_host_control import ControlHost,SymcoreHost
ROW={"hr":"8","weekday":"1","workingday":"1","weathersit":"1"}
def test_real_hosts_are_causal_and_learn():
 c=ControlHost();s=SymcoreHost()
 a=c.step(ROW,100);b=s.step(ROW,100)
 assert a==100 and b==100
 assert c.step(ROW,100)<a and s.step(ROW,100)<=b

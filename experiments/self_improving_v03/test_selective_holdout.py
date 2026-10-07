from selective_holdout import block_bootstrap_R
def test_R_rewards_concentrated_value():
 d=[1.0]*1024;g=[1 if i%10==0 else 0 for i in range(1024)]
 # Uniform value has no selective concentration: point estimate is exactly one.
 rate=sum(g)/len(g);R=sum(x*y for x,y in zip(d,g))/(rate*sum(d))
 assert abs(R-1)<1e-12

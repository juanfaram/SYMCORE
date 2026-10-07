import math
from real_cost_meter import CostMeter,normalized_vector
def work(n):
 x=0
 for i in range(n):x+=i*i
 return x
def test_cost_is_finite_without_any_advantage_variable():
 m=CostMeter()
 for _ in range(5):m.measure(work,1000,updates=4)
 v=normalized_vector(m.snapshot())
 assert all(math.isfinite(float(x)) for x in v.values()) and v["updates_per_interaction"]==4
def test_cost_records_more_algorithmic_work_without_dividing_by_value():
 a=CostMeter();b=CostMeter()
 for _ in range(3):a.measure(work,100,updates=1);b.measure(work,100,updates=7)
 assert normalized_vector(b.snapshot())["updates_per_interaction"]>normalized_vector(a.snapshot())["updates_per_interaction"]
def test_zero_updates_is_valid_and_finite():
 m=CostMeter();m.measure(work,10,updates=0);v=normalized_vector(m.snapshot())
 assert v["updates_per_interaction"]==0 and math.isfinite(v["seconds_per_interaction"])

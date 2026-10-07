from acquisition_metric import acquisition_cost
from tabular_capability_factory import TabularCapabilityFactory
def test_metric_distinguishes_learning_curves():
 assert acquisition_cost([10]*50)["cost"]>acquisition_cost([10,8,6,4,2]+[1]*45)["cost"]
def test_tabular_factory_generates_nonredundant_specs():
 xs=TabularCapabilityFactory().propose(4,{0:.9,1:.7,2:.2,3:.1},3,5)
 assert len({x["features"] for x in xs})==len(xs) and len(xs)>1

import math,random
from functional_geometry import FunctionalGeometry,distance
def feed(g,names,scale,offset,seed=1):
 r=random.Random(seed)
 for t in range(80):
  base=[math.sin(t/9)+.015*t, math.cos(t/13)+(.8 if t>55 else 0), .3*math.sin(t/4)+r.gauss(0,.08)]
  g.observe({names[i]:offset[i]+scale[i]*base[i] for i in range(3)})
def test_geometry_invariant_to_sensor_names_order_and_affine_scale():
 a=FunctionalGeometry();b=FunctionalGeometry()
 feed(a,["error","entropy","load"],[1,1,1],[0,0,0])
 feed(b,["foo","bar","baz"],[7,.2,3],[10,-4,100])
 assert distance(a.vector(),b.vector())<1e-9
def test_geometry_separates_different_dynamics():
 a=FunctionalGeometry();b=FunctionalGeometry()
 feed(a,["a","b","c"],[1,1,1],[0,0,0])
 for t in range(80):b.observe({"x":0.1*t*t,"y":(-1)**t,"z":0.0})
 assert distance(a.vector(),b.vector())>.25
def test_geometry_fixed_dimension_with_different_channel_counts():
 a=FunctionalGeometry();b=FunctionalGeometry()
 for t in range(20):a.observe({"x":t});b.observe({"u":t,"v":2*t,"w":3*t})
 assert len(a.vector())==len(b.vector())==15

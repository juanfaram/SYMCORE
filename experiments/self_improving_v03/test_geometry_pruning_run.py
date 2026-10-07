from geometry_necessity_destruction_run import numeric_signals
def test_target_never_enters_geometry():
 x=numeric_signals({"a":"1.5","__target":999,"name":"x"})
 assert x=={"a":1.5}

def test_geometry_accepts_numpy_like_scalars_as_native_floats():
 from functional_geometry import FunctionalGeometry
 try:
  import numpy as np
  vals=[np.float64(i) for i in range(8)]
 except ImportError:
  vals=[float(i) for i in range(8)]
 g=FunctionalGeometry()
 for v in vals:g.observe({"x":v})
 z=g.vector()
 assert len(z)==15 and all(isinstance(x,float) for x in z)

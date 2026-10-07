from geometry_necessity_destruction_run import numeric_signals
def test_target_never_enters_geometry():
 x=numeric_signals({"a":"1.5","__target":999,"name":"x"})
 assert x=={"a":1.5}

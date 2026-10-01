"""Calibration reformulation: repeated linear membership -> indexed set membership."""
def linear_membership(values, queries):
    return [any(v == q for v in values) for q in queries]

def omega_membership(values, queries):
    index = set(values)
    return [q in index for q in queries]

def hostile_membership(values, queries):
    # Human/reference specialist baseline: direct idiomatic Python set solution.
    s = set(values)
    return [q in s for q in queries]

"""Ω minimum corpus v1: five semantically distinct optimization problems."""

def reduction_sum(xs):
    total = 0
    for x in xs:
        total += x
    return total

def map_filter_fusion(xs):
    ys = [x * 2 for x in xs]
    return [y for y in ys if y % 3 == 0]

def repeated_membership(xs, queries):
    out = []
    for q in queries:
        out.append(q in xs)
    return out

def prefix_sum(xs):
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out

def repeated_subproblem(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

"""Reference semantic candidates. These are hypotheses until verified by runner.py."""

def reduction_sum(xs):
    return sum(xs)

def map_filter_fusion(xs):
    return [x * 2 for x in xs if (x * 2) % 3 == 0]

def repeated_membership(xs, queries):
    lookup = set(xs)
    return [q in lookup for q in queries]

def prefix_sum(xs):
    import itertools
    return list(itertools.accumulate(xs))

def repeated_subproblem(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

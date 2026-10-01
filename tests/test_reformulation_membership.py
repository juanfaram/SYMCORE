import random
from omega.reformulation.membership import linear_membership, omega_membership, hostile_membership

def test_examples_equivalent():
    cases=[
        ([],[]),([], [1]),([1],[1,2]),([1,1,2],[1,2,3]),
        ([-2,0,3],[-2,2,3]),(["a","b","a"],["a","c"])
    ]
    for values,queries in cases:
        expected=linear_membership(values,queries)
        assert omega_membership(values,queries)==expected
        assert hostile_membership(values,queries)==expected

def test_random_integer_equivalence():
    rng=random.Random(3302)
    for _ in range(500):
        n=rng.randrange(0,80); qn=rng.randrange(0,80)
        values=[rng.randrange(-50,51) for _ in range(n)]
        queries=[rng.randrange(-70,71) for _ in range(qn)]
        expected=linear_membership(values,queries)
        assert omega_membership(values,queries)==expected
        assert hostile_membership(values,queries)==expected

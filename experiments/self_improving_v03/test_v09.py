from repertoire import Repertoire,Evidence
from examiner import Examiner
def test_repertoire_preserves_distinct_niches():
 r=Repertoire();a=Evidence(.8,.9,.4,.8,.8,.7);b=Evidence(.8,.4,.9,.8,.8,.7)
 assert r.consider("fast-adapt",a) and r.consider("retainer",b) and len(r.cells)==2
def test_repertoire_replaces_weaker_same_niche():
 r=Repertoire();a=Evidence(.5,.8,.8,.8,.8,.2);b=Evidence(.9,.81,.81,.81,.81,.2)
 r.consider("a",a);assert r.consider("b",b);assert any(n=="b" for n,_ in r.cells.values())
def test_examiner_targets_observed_weakness():
 e=Examiner(1)
 for _ in range(100):e.observe({"domain":"code","difficulty":"hard"},False)
 for _ in range(100):e.observe({"domain":"writing","difficulty":"simple"},True)
 c=e.next_context();assert c["domain"]=="code" and c["difficulty"]=="hard"

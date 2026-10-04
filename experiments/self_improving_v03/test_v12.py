from recombination import Skill,Composer
from procedural_examiner import ProceduralExaminer
def test_recombination_expands_strengths():
 a=Skill("a",frozenset(["x"]),{"adapt":.9},1);b=Skill("b",frozenset(["y"]),{"retain":.9},1);c=Composer().compose(a,b)
 assert c.strengths["adapt"]==.9 and c.strengths["retain"]==.9 and c.domains==frozenset(["x","y"])
def test_examiner_targets_unmastered_space():
 e=ProceduralExaminer(1)
 for _ in range(150):e.observe({"domain":"code","difficulty":"hard"},True)
 c=e.generate();assert not (c["domain"]=="code" and c["difficulty"]=="hard" and c["perturbation"]=="base")

from pathlib import Path

def test_python_sources_do_not_contain_literal_backslash_n_code():
    root=Path(__file__).parent
    offenders=[]
    for p in root.glob("*.py"):
        if p.name==Path(__file__).name:continue
        for i,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
            # Recurrent corruption pattern: a literal escape inserted between Python statements.
            if "\\n   " in line or "\\n def " in line or "\\n out=" in line:
                offenders.append(f"{p.name}:{i}")
    assert not offenders, "literal \\n code escapes found: "+", ".join(offenders)

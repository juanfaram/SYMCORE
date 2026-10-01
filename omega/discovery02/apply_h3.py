from pathlib import Path
p=Path("Lib/_pybuiltins.py")
s=p.read_text()
old="""    cls = type(async_iterator)
    try:
        # Looked up on the type, like the C slot am_anext.
        anext_method = cls.__anext__
    except AttributeError:
        raise TypeError(
            f"{cls.__name__!r} object is not an async iterator"
        ) from None
    awaitable = anext_method(async_iterator)
"""
new="""    try:
        # H3: preserve the Python frame/default semantics while avoiding
        # repeated explicit type() + class attribute lookup + unbound call.
        awaitable = async_iterator.__anext__()
    except AttributeError:
        cls = type(async_iterator)
        raise TypeError(
            f"{cls.__name__!r} object is not an async iterator"
        ) from None
"""
if old not in s:
    raise SystemExit("expected source block not found")
p.write_text(s.replace(old,new))

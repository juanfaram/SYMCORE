from pathlib import Path
cpath=Path("Python/bltinmodule.c")
c=cpath.read_text()
marker="static PyMethodDef builtin_methods[] = {"
helper=r'''
static PyObject *
omega_anext_slot(PyObject *module, PyObject *aiterator)
{
    PyTypeObject *t = Py_TYPE(aiterator);
    if (t->tp_as_async == NULL || t->tp_as_async->am_anext == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "'%.200s' object is not an async iterator",
                     t->tp_name);
        return NULL;
    }
    return (*t->tp_as_async->am_anext)(aiterator);
}

'''
if helper not in c:
    c=c.replace(marker,helper+marker)
entry='    {"_anext_slot", omega_anext_slot, METH_O, NULL},\n'
if entry not in c:
    c=c.replace(marker,marker+"\n"+entry)
cpath.write_text(c)

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
new="""    # Hybrid: preserve this Python frame/default semantics, but delegate
    # special-method slot dispatch to a private C helper.
    awaitable = _anext_slot(async_iterator)
"""
if old not in s:
    raise SystemExit("expected _pybuiltins block not found")
p.write_text(s.replace(old,new))

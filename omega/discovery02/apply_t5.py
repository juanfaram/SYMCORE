from pathlib import Path
p=Path("Lib/_pybuiltins.py")
s=p.read_text()
s=s.replace("__all__ = ['anext']", "__all__ = []")
s=s.replace("for _name in __all__:\n    globals()[_name].__module__ = 'builtins'\ndel _name", "# T5: no public Python builtin copied from this module")
start=s.index("def anext(async_iterator, default=_NOT_GIVEN, /):")
end=s.index("\n\nasync def _anext_with_default", start)
replacement='''def _anext_default_entry(async_iterator, default, /):
    """Private Python path used by the C dispatcher for the observable default case."""
    cls = type(async_iterator)
    try:
        anext_method = cls.__anext__
    except AttributeError:
        raise TypeError(
            f"{cls.__name__!r} object is not an async iterator"
        ) from None
    awaitable = anext_method(async_iterator)
    return _anext_with_default(awaitable, default)
'''
s=s[:start]+replacement+s[end:]
p.write_text(s)

cpath=Path("Python/bltinmodule.c")
c=cpath.read_text()
marker="static PyMethodDef builtin_methods[] = {"
helper=r'''
static PyObject *
omega_builtin_anext(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (!_PyArg_CheckPositional("anext", nargs, 1, 2)) {
        return NULL;
    }
    PyObject *aiterator = args[0];
    if (nargs == 1) {
        PyTypeObject *t = Py_TYPE(aiterator);
        if (t->tp_as_async == NULL || t->tp_as_async->am_anext == NULL) {
            PyErr_Format(PyExc_TypeError,
                         "'%.200s' object is not an async iterator",
                         t->tp_name);
            return NULL;
        }
        return (*t->tp_as_async->am_anext)(aiterator);
    }

    PyObject *mod = PyImport_AddModuleRef("_pybuiltins");
    if (mod == NULL) {
        return NULL;
    }
    PyObject *func = PyObject_GetAttrString(mod, "_anext_default_entry");
    Py_DECREF(mod);
    if (func == NULL) {
        return NULL;
    }
    PyObject *res = PyObject_CallFunctionObjArgs(func, aiterator, args[1], NULL);
    Py_DECREF(func);
    return res;
}

PyDoc_STRVAR(omega_builtin_anext_doc,
"anext($module, async_iterator, default=<unrepresentable>, /)\n"
"--\n\n"
"Return the next item from the async iterator.\n\n"
"If default is given and the async iterator is exhausted,\n"
"it is returned instead of raising StopAsyncIteration.");

'''
if helper not in c:
    c=c.replace(marker,helper+marker)
entry='    {"anext", _PyCFunction_CAST(omega_builtin_anext), METH_FASTCALL, omega_builtin_anext_doc},\n'
if entry not in c:
    c=c.replace(marker,marker+"\n"+entry)
cpath.write_text(c)

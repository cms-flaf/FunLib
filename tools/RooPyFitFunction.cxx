/**
 * RooPyFitFunction.cxx  —  implementation
 *
 * See RooPyFitFunction.h for design notes.
 */

/* Python.h first — avoids macro-redefinition warnings on some platforms. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "RooPyFitFunction.h"
#include "RooArgList.h"
#include "RooAbsReal.h"

#include <cmath>
#include <iostream>

ClassImp(RooPyFitFunction)

/* ── constructors / destructor ──────────────────────────────────────────────── */

RooPyFitFunction::RooPyFitFunction()
    : RooAbsPdf(),
      _x(),
      _params(),
      _pyKernel(nullptr),
      _pyInited(false)
{}

RooPyFitFunction::RooPyFitFunction(
        const char*       name,
        const char*       title,
        RooAbsReal&       x,
        const RooArgList& params,
        const char*       meta_json,
        const char*       repo_root)
    : RooAbsPdf(name, title),
      _x("_x", "observable", this, x),
      _params("_params", "shape params", this),
      _metaJson(meta_json),
      _repoRoot(repo_root),
      _pyKernel(nullptr),
      _pyInited(false)
{
    _params.add(params);
}

RooPyFitFunction::RooPyFitFunction(const RooPyFitFunction& other, const char* name)
    : RooAbsPdf(other, name),
      _x("_x", this, other._x),
      _params("_params", this, other._params),
      _metaJson(other._metaJson),
      _repoRoot(other._repoRoot),
      _pyKernel(nullptr),   /* transient: NOT copied; re-init on first evaluate() */
      _pyInited(false)
{}

RooPyFitFunction::~RooPyFitFunction()
{
    if (_pyKernel && Py_IsInitialized()) {
        PyGILState_STATE g = PyGILState_Ensure();
        Py_XDECREF(_pyKernel);
        _pyKernel = nullptr;
        PyGILState_Release(g);
    }
}

/* ── kernel initialisation ──────────────────────────────────────────────────── */

bool RooPyFitFunction::_ensureKernel() const
{
    /* Guard: only try once. If _pyKernel is null after init, give up. */
    if (_pyInited) return (_pyKernel != nullptr);
    _pyInited = true;

    /* ---- 1. Ensure Python interpreter is running --------------------- */
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "[RooPyFitFunction] ERROR: failed to initialize Python\n";
            return false;
        }
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    /* ---- 2. Prepend repo root to sys.path ---------------------------- */
    {
        PyObject* sys  = PyImport_ImportModule("sys");
        PyObject* path = PyObject_GetAttrString(sys, "path");
        PyObject* root = PyUnicode_FromString(_repoRoot.Data());
        if (PySequence_Contains(path, root) == 0)
            PyList_Insert(path, 0, root);
        Py_DECREF(root);
        Py_DECREF(path);
        Py_DECREF(sys);
    }

    /* ---- 3. Parse fn_meta JSON --------------------------------------- */
    PyObject* jsonMod = PyImport_ImportModule("json");
    if (!jsonMod) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return false;
    }
    PyObject* metaStr  = PyUnicode_FromString(_metaJson.Data());
    PyObject* loadsFn  = PyObject_GetAttrString(jsonMod, "loads");
    PyObject* loadArgs = PyTuple_Pack(1, metaStr);
    PyObject* metaDict = PyObject_Call(loadsFn, loadArgs, nullptr);
    Py_DECREF(loadArgs);
    Py_DECREF(metaStr);
    Py_DECREF(loadsFn);
    Py_DECREF(jsonMod);

    if (!metaDict) {
        std::cerr << "[RooPyFitFunction] ERROR: failed to parse meta JSON\n";
        PyErr_Print();
        PyGILState_Release(gstate);
        return false;
    }

    /* Extract entry sub-dict, xmin, xmax */
    PyObject* entryDict = PyDict_GetItemString(metaDict, "entry");  /* borrowed */
    PyObject* xminObj   = PyDict_GetItemString(metaDict, "xmin");
    PyObject* xmaxObj   = PyDict_GetItemString(metaDict, "xmax");

    if (!entryDict) {
        std::cerr << "[RooPyFitFunction] ERROR: 'entry' missing from meta JSON\n";
        Py_DECREF(metaDict);
        PyGILState_Release(gstate);
        return false;
    }

    /* ---- 4. Import load_kernel from FunLib.tools.factory ------------- */
    PyObject* factory = PyImport_ImportModule("FunLib.tools.factory");
    if (!factory) {
        std::cerr << "[RooPyFitFunction] ERROR: cannot import FunLib.tools.factory\n";
        PyErr_Print();
        Py_DECREF(metaDict);
        PyGILState_Release(gstate);
        return false;
    }
    PyObject* loadKernel = PyObject_GetAttrString(factory, "load_kernel");
    Py_DECREF(factory);
    if (!loadKernel) {
        PyErr_Print();
        Py_DECREF(metaDict);
        PyGILState_Release(gstate);
        return false;
    }

    /* ---- 5. Call load_kernel(entry, repo_root=…, xmin=…, xmax=…) ---- */
    PyObject* kw = PyDict_New();
    PyDict_SetItemString(kw, "repo_root", PyUnicode_FromString(_repoRoot.Data()));
    if (xminObj) PyDict_SetItemString(kw, "xmin", xminObj);
    if (xmaxObj) PyDict_SetItemString(kw, "xmax", xmaxObj);

    PyObject* posArgs = PyTuple_Pack(1, entryDict);
    _pyKernel = PyObject_Call(loadKernel, posArgs, kw);
    Py_DECREF(posArgs);
    Py_DECREF(kw);
    Py_DECREF(loadKernel);
    Py_DECREF(metaDict);

    if (!_pyKernel) {
        std::cerr << "[RooPyFitFunction] ERROR: load_kernel() failed\n";
        PyErr_Print();
        PyGILState_Release(gstate);
        return false;
    }

    /* ---- 6. Cache kernel.param_names --------------------------------- */
    PyObject* pnames = PyObject_GetAttrString(_pyKernel, "param_names");
    if (pnames) {
        Py_ssize_t n = PySequence_Length(pnames);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* s = PySequence_GetItem(pnames, i);  /* new ref */
            _paramNames.push_back(PyUnicode_AsUTF8(s));
            Py_DECREF(s);
        }
        Py_DECREF(pnames);
    }

    PyGILState_Release(gstate);
    return true;
}

/* ── evaluate ───────────────────────────────────────────────────────────────── */

Double_t RooPyFitFunction::evaluate() const
{
    if (!_ensureKernel()) return 0.0;

    PyGILState_STATE gstate = PyGILState_Ensure();

    /* Build keyword-argument dict from current RooRealVar values. */
    int nPar = static_cast<int>(_params.size());
    PyObject* kwargs = PyDict_New();
    for (int i = 0; i < nPar && i < static_cast<int>(_paramNames.size()); i++) {
        const RooAbsReal* par = static_cast<const RooAbsReal*>(_params.at(i));
        PyObject* v = PyFloat_FromDouble(par->getVal());
        PyDict_SetItemString(kwargs, _paramNames[i].c_str(), v);
        Py_DECREF(v);
    }

    /* Call: kernel(x_val, **kwargs)  →  pure shape value (no N) */
    PyObject* xArg  = PyFloat_FromDouble(double(_x));
    PyObject* args  = PyTuple_Pack(1, xArg);
    PyObject* res   = PyObject_Call(_pyKernel, args, kwargs);
    Py_DECREF(xArg);
    Py_DECREF(args);
    Py_DECREF(kwargs);

    double ret = 0.0;
    if (res) {
        ret = PyFloat_AsDouble(res);
        Py_DECREF(res);
    } else {
        PyErr_Clear();
    }
    /* Guard: NaN / negative / inf → 0 (same contract as Python kernels) */
    if (!std::isfinite(ret) || ret < 0.0) ret = 0.0;

    PyGILState_Release(gstate);
    return ret;
}

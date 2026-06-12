"""load_workspace -- Load a RooFit workspace written by mk_workspace."""

from __future__ import annotations

import json

__all__ = ["WorkspaceHandle", "load_workspace"]


class WorkspaceHandle:
    """
    Container for a loaded RooFit workspace.

    The backing TFile is kept open so that RooFit objects (workspace, data,
    variables) remain valid.  Use as a context manager for automatic cleanup:

        with load_workspace("result.root") as wh:
            handle = wh.make_handle()
            ext_pdf, N_roo, shape_rvs = handle.make_roo_ext_pdf(wh.x_obs)
            ext_pdf.fitTo(wh.data, ...)

    Attributes
    ----------
    ws   : ROOT.RooWorkspace -- the loaded workspace
    meta : dict              -- JSON metadata written by mk_workspace
    """

    def __init__(self, tfile, ws, meta: dict):
        self._file = tfile  # keep alive: closing it would invalidate ws objects
        self.ws = ws
        self.meta = meta  # plain Python dict -- no ROOT dependency

    # -- convenience properties ------------------------------------------------

    @property
    def x_obs(self):
        """Observable RooRealVar (``x``)."""
        return self.ws.var("x")

    @property
    def data(self):
        """RooDataHist (``data_obs`` -- the observed data)."""
        return self.ws.data("data_obs")

    @property
    def model(self):
        """Signal+background RooAddPdf (``model_s``)."""
        return self.ws.pdf("model_s")

    @property
    def entry(self) -> dict:
        """Function entry dict (as written to the workspace by mk_workspace)."""
        return self.meta["entry"]

    @property
    def excludes(self) -> list:
        """Blinded windows as list of (lo, hi) tuples."""
        return [tuple(p) for p in self.meta["excludes"]]

    @property
    def key(self) -> str:
        """Resolved function key (e.g. ``'BWZxBern-1'``)."""
        return self.meta["key"]

    @property
    def param_names(self) -> list:
        """Shape parameter names (no N)."""
        return self.meta["param_names"]

    # -- parameter access ------------------------------------------------------

    def get_param(self, name: str):
        """
        Return the RooRealVar for shape parameter *name*.

        Parameters are stored in the workspace as ``p_{name}`` to avoid
        conflicts with ROOT reserved names.  Returns None if not found.
        """
        return self.ws.var(f"p_{name}")

    def current_params(self) -> list:
        """
        Return current RooRealVar values for all shape parameters.

        Useful after fitting to read off fitted values:

            handle.initial_params = wh.current_params()
        """
        return [self.get_param(n).getVal() for n in self.param_names]

    # -- handle reconstruction -------------------------------------------------

    def make_handle(self, *, repo_root: str | None = None):
        """
        Reconstruct a :class:`~FunLib.tools.factory.FunctionHandle` from
        workspace metadata.

        The handle's ``initial_params`` are taken from the metadata stored in
        the workspace (the values passed to ``mk_workspace``).  To warm-start
        from previously fitted workspace parameter values instead:

            wh.entry["initial_params"] = wh.current_params()
            handle = wh.make_handle()

        Parameters
        ----------
        repo_root : str, optional
            Repo root for resolving ``entry["dir"]``. Auto-detected if None.

        Returns
        -------
        FunctionHandle
        """
        from FunLib.tools.factory import make_function

        return make_function(
            self.entry,
            xmin=self.meta["xmin"],
            xmax=self.meta["xmax"],
            excludes=self.excludes,
            bin_width=self.meta["bin_width"],
            integral=self.meta["integral"],
            repo_root=repo_root,
        )

    # -- context manager -------------------------------------------------------

    def close(self):
        """Close the backing TFile. Workspace objects become invalid after this."""
        if self._file is not None:
            try:
                if self._file.IsOpen():
                    self._file.Close()
            except Exception:
                pass
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return (
            f"WorkspaceHandle(key={self.meta.get('key')!r}, "
            f"ws={self.ws.GetName()!r})"
        )


# -- load_workspace ------------------------------------------------------------


def load_workspace(root_file: str, *, ws_name: str = "w") -> WorkspaceHandle:
    """
    Load a RooFit workspace written by :func:`~FunLib.tools.mk_workspace.mk_workspace`.

    Parameters
    ----------
    root_file : str
        Path to the .root file produced by ``mk_workspace``.
    ws_name : str
        Workspace name inside the file. Default: ``"w"``.

    Returns
    -------
    WorkspaceHandle
        The backing TFile is kept open.  Call ``wh.close()`` when done, or
        use the returned object as a context manager (``with`` statement).

    Raises
    ------
    FileNotFoundError
        If *root_file* cannot be opened.
    KeyError
        If the workspace or ``fn_meta`` object is missing from the file.
    """
    import ROOT

    tf = ROOT.TFile.Open(root_file, "READ")
    if not tf or tf.IsZombie():
        raise FileNotFoundError(f"Cannot open ROOT file: {root_file!r}")

    ws = tf.Get(ws_name)
    if not ws:
        tf.Close()
        raise KeyError(f"Workspace {ws_name!r} not found in {root_file!r}")

    meta_obj = tf.Get("fn_meta")
    if not meta_obj:
        tf.Close()
        raise KeyError(
            f"'fn_meta' metadata not found in {root_file!r} "
            f"(was this file written by mk_workspace?)"
        )

    meta = json.loads(meta_obj.GetString().Data())
    return WorkspaceHandle(tf, ws, meta)

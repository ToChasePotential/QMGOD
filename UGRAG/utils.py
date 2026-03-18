# chugao/utils.py
# =====================================================
# Shared utilities for UnionGraph-RAG pipeline
# =====================================================

import inspect


def _info(msg: str):
    print(f"[INFO] {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


# -----------------------------------------------------
# Resolve function by name (safe)
# -----------------------------------------------------
def resolve_func(module, candidates):
    """
    Try to find a callable in module by candidate names.
    """
    for name in candidates:
        if hasattr(module, name):
            fn = getattr(module, name)
            if callable(fn):
                return fn
    return None


# -----------------------------------------------------
# Safe call with auto keyword filtering
# -----------------------------------------------------
def call_safely(fn, **kwargs):
    """
    Call fn with only accepted keyword arguments.
    """
    sig = inspect.signature(fn)
    accepted = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
    return fn(**accepted)
    
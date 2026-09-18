#!/bin/bash
#----------------------------------------------------------------------------
# Shared environment preparation for every LS6 script. Source it, do not run it.
#
# Fixes the failure that killed baseline job 3452729:
#
#   ImportError: cannot load module more than once per process
#     numpy/_core/multiarray.py -> _multiarray_umath
#
# That error means numpy's C extension was loaded TWICE in one process, under two
# different paths. On LS6 there are two ways that happens and both are the
# surrounding environment leaking into the venv:
#
#  1. TACC loads python3/3.9.7 by DEFAULT (it is in `module list` on a fresh
#     login) and its modulefile puts entries on PYTHONPATH. Those stay on
#     sys.path even after the venv is activated, because PYTHONPATH is searched
#     before site-packages -- so a second numpy can be found there.
#
#  2. Anything previously pip-installed with --user lands in
#     ~/.local/lib/python3.12/site-packages, which is also on sys.path ahead of
#     the venv unless PYTHONNOUSERSITE is set.
#
# Clearing both leaves exactly one numpy on the path, which is the only
# configuration that works.
#----------------------------------------------------------------------------

module unload python3 2>/dev/null || true      # the default-loaded 3.9.7
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
module load python/3.12.11 2>/dev/null || true

unset PYTHONPATH                # see (1)
export PYTHONNOUSERSITE=1       # see (2)
export PYTHONUNBUFFERED=1

ls6_env_report() {
    echo "  python : $(command -v python)"
    python - <<'PY'
import sys, collections
print(f"  version: {sys.version.split()[0]}")
print(f"  prefix : {sys.prefix}")
dupes = [p for p, n in collections.Counter(sys.path).items() if n > 1]
if dupes:
    print(f"  !! DUPLICATE sys.path entries: {dupes}")
try:
    import numpy
    print(f"  numpy  : {numpy.__version__} from {numpy.__file__}")
except Exception as e:
    print(f"  numpy  : FAILED -- {type(e).__name__}: {e}")
PY
}

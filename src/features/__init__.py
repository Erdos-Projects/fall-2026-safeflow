"""Public feature API backed by the byte-preserved research implementation.

The historical module stays at its original location so recorded source hashes,
dataset fingerprints and archived notebooks remain valid. Import ``shared`` here
in new code; do not duplicate feature formulas.
"""
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).resolve().parents[2] / "notebooks" / "safeflow_features.py"
if "safeflow_features" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("safeflow_features", _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
shared = sys.modules["safeflow_features"]

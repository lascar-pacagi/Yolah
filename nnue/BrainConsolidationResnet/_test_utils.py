"""Helpers shared by every exercise's test.py.

The exercise files are not part of a Python package — they live in
exercises/NN_name/exercise.py and we want each test to load *its own*
sibling exercise.py without polluting sys.modules.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
CACHE_DIR = FIXTURES / "cache"


def load_exercise(test_file_path):
    """
    Import the `exercise.py` next to the given test file as a fresh module.

    Each call returns a NEW module object even if cached, so tests across
    multiple exercises don't leak state into each other.
    """
    test_path = Path(test_file_path).resolve()
    ex_path = test_path.parent / "exercise.py"
    if not ex_path.exists():
        raise FileNotFoundError(f"No exercise.py next to {test_path}")

    # Unique module name per exercise dir, so reloads don't clobber.
    mod_name = f"exercise_{test_path.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, ex_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

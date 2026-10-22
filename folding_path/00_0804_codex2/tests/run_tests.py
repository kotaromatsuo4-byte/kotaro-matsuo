"""Run every qd_folding test without requiring pytest."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import unittest


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Add both the project and its parent so the runner works from any current
# working directory.
for folder in (WORKSPACE_ROOT, PROJECT_ROOT):
    value = str(folder)
    if value not in sys.path:
        sys.path.insert(0, value)
sys.dont_write_bytecode = True


def build_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for path in sorted(TEST_DIR.glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))

        # unittest intentionally ignores pytest-style module functions.  The
        # mechanics tests use only plain asserts, so wrap their zero-argument
        # functions in FunctionTestCase rather than adding a pytest dependency.
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            if inspect.signature(function).parameters:
                continue
            suite.addTest(
                unittest.FunctionTestCase(
                    function,
                    description=f"{path.stem}.{name}",
                )
            )
    return suite


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(build_suite())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

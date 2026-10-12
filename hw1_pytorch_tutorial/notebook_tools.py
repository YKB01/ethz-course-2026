"""Small standard-library helpers for checking and executing notebook definitions."""

import ast
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_definitions(name):
    """Load imports/classes/functions without downloading data or starting training."""
    notebook = json.loads((ROOT / "src" / f"{name}.ipynb").read_text())
    module = types.ModuleType(f"hw1_{name}")
    sys.modules[module.__name__] = module
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            tree.body = [node for node in tree.body if isinstance(node, allowed)]
            exec(compile(tree, f"{name}.ipynb", "exec"), module.__dict__)
    return module

"""Machine-checkable parts of the PDF §4 engineering rules (M6b compliance audit).

* "Every public model function has type hints, docstrings, explicit units and a short reference
  to the governing equation or standard": type hints are enforced by ``mypy --strict`` on
  ``src``; this module enforces that every public module, class, function and method has a
  docstring. Units and references are reviewed by hand (``docs/compliance_matrix.md``).
* "Avoid global mutable state": no module of ``src/helioswarm`` uses the ``global`` statement;
  no module-level name is bound to a mutable container (``list``/``dict``/``set`` literal,
  comprehension or constructor, ``bytearray``) or to a NumPy array constructor
  (``np.array``/``zeros``/... ; a module-level array is accepted only if the module makes it
  read-only with ``.setflags(write=False)``) except the dunder ``__all__``; and no function or
  method is wrapped in a memoisation cache (``functools.lru_cache``/``cache``), which is
  hidden process-wide state. Read-only tables use ``types.MappingProxyType`` or tuples.
  (Class-attribute configuration of the private YAML loader in ``config.py``, set once at
  import as PyYAML requires, is not a name binding and is outside this check; mutation of a
  module-level object through a method call is not detectable statically and is reviewed by
  hand.)
* No stub modules for later milestones (PDF §15; plan M6b): ``scheduler``, ``optimization`` and
  ``paper/`` do not exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "helioswarm"
REPO = SRC.parents[1]
MODULES = sorted(SRC.glob("*.py"))


def _public_defs(tree: ast.Module) -> list[tuple[int, str, ast.AST]]:
    """Public classes, top-level functions and methods of public classes (not nested local
    helper functions, which are implementation details of their enclosing function)."""
    out: list[tuple[int, str, ast.AST]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if child.name.startswith("_"):
                    continue
                out.append((child.lineno, prefix + child.name, child))
                if isinstance(child, ast.ClassDef):
                    visit(child, prefix + child.name + ".")

    visit(tree, "")
    return out


def _missing_docstrings(source: str) -> list[str]:
    tree = ast.parse(source)
    missing = [] if ast.get_docstring(tree) else ["<module>"]
    for lineno, name, node in _public_defs(tree):
        assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        if ast.get_docstring(node) is None:
            missing.append(f"{name} (line {lineno})")
    return missing


def test_modules_exist() -> None:
    assert len(MODULES) >= 30  # the M0-M6 package (a wrong SRC path would find nothing)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_public_definition_has_a_docstring(path: Path) -> None:
    assert _missing_docstrings(path.read_text(encoding="utf-8")) == []


def test_docstring_check_is_discriminating() -> None:
    src = '"""Mod."""\n\ndef f() -> None:\n    pass\n\nclass C:\n    """C."""\n\n' + (
        "    def m(self) -> None:\n        pass\n\n    def _private(self) -> None:\n        pass\n"
    )
    assert _missing_docstrings(src) == ["f (line 3)", "C.m (line 9)"]
    assert _missing_docstrings("def g() -> None:\n    '''G.'''\n") == ["<module>"]


_MUTABLE_CALLS = frozenset(
    {"list", "dict", "set", "bytearray", "defaultdict", "OrderedDict", "Counter", "deque"}
)
_NUMPY_ARRAY_CALLS = frozenset(
    {
        "array", "asarray", "asanyarray", "ascontiguousarray", "zeros", "ones", "empty", "full",
        "zeros_like", "ones_like", "empty_like", "full_like", "eye", "identity", "arange",
        "linspace", "logspace", "geomspace", "diag", "frombuffer", "fromiter", "concatenate",
        "stack", "column_stack", "vstack", "hstack", "copy",
    }
)  # fmt: skip
_NUMPY_ALIASES = frozenset({"np", "numpy"})
_CACHE_DECORATORS = frozenset({"lru_cache", "cache"})


def _call_name(node: ast.expr) -> str:
    """Dotted name of a call target (``np.zeros`` -> ``np.zeros``), or ``""``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _is_numpy_array_call(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    name = _call_name(value.func)
    head, _, tail = name.rpartition(".")
    return head in _NUMPY_ALIASES and tail in _NUMPY_ARRAY_CALLS


def _made_read_only(tree: ast.Module) -> set[str]:
    """Module-level names on which ``<name>.setflags(write=False)`` is called at module level."""
    out: set[str] = set()
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        f = stmt.value.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "setflags"
            and isinstance(f.value, ast.Name)
            and any(
                k.arg == "write" and isinstance(k.value, ast.Constant) and k.value.value is False
                for k in stmt.value.keywords
            )
        ):
            out.add(f.value.id)
    return out


def _module_level_mutables(source: str) -> list[str]:
    tree = ast.parse(source)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            bad.append(f"global statement (line {node.lineno})")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if _call_name(target).rpartition(".")[2] in _CACHE_DECORATORS:
                    bad.append(f"memoisation cache on {node.name} (line {node.lineno})")
    read_only = _made_read_only(tree)
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        if value is None:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names or names == ["__all__"]:
            continue
        mutable = isinstance(
            value, ast.List | ast.Dict | ast.Set | ast.ListComp | ast.DictComp | ast.SetComp
        )
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            mutable = mutable or value.func.id in _MUTABLE_CALLS
        if _is_numpy_array_call(value) and not set(names) <= read_only:
            mutable = True
        if mutable:
            bad.append(f"{', '.join(names)} (line {stmt.lineno})")
    return bad


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_global_mutable_state(path: Path) -> None:
    assert _module_level_mutables(path.read_text(encoding="utf-8")) == []


def test_mutable_state_check_is_discriminating() -> None:
    src = "CACHE = {}\nX: list[int] = []\nY = dict()\nZ = (1, 2)\n__all__ = ['a']\n"
    src += "def f() -> None:\n    global Z\n"
    assert _module_level_mutables(src) == [
        "global statement (line 7)",
        "CACHE (line 1)",
        "X (line 2)",
        "Y (line 3)",
    ]
    # NumPy arrays, set comprehensions, bytearray and memoisation caches (M6b review).
    src = (
        "import functools\nimport numpy as np\n"
        "R = np.zeros(3)\n"  # line 3
        "S = {x for x in range(3)}\n"  # line 4
        "B = bytearray(4)\n"  # line 5
        "@functools.lru_cache\ndef f(x):\n    return x\n"  # def on line 7
        "@functools.lru_cache(maxsize=8)\ndef g(x):\n    return x\n"  # def on line 10
        "from functools import cache\n@cache\ndef h(x):\n    return x\n"  # def on line 14
        "class K:\n    @functools.cache\n    def m(self):\n        return 1\n"  # line 18
        "RO = np.array([1.0, 2.0])\nRO.setflags(write=False)\n"  # read-only: accepted
        "T = np.asarray((1.0,)).sum()\n"  # a scalar, not an array constructor: accepted
    )
    assert _module_level_mutables(src) == [
        "memoisation cache on f (line 7)",
        "memoisation cache on g (line 10)",
        "memoisation cache on h (line 14)",
        "memoisation cache on m (line 18)",
        "R (line 3)",
        "S (line 4)",
        "B (line 5)",
    ]


def test_no_stubs_for_later_milestones() -> None:
    for name in ("scheduler.py", "optimization.py"):
        assert not (SRC / name).exists(), name
    assert not (REPO / "paper").exists()

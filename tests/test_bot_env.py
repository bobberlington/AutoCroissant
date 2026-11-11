"""The bot's environment (requirements-bot.txt, the jetson's venv): main.py's import closure, lazy imports included,
never reaches the card toolchain, and every package it imports is pinned there."""
import ast
import re
import sys
from importlib.metadata import packages_distributions

from conftest import REPO

TOOLCHAIN = {"psd_tools", "scipy", "fontTools", "uharfbuzz", "freetype", "aggdraw", "Cython", "cython", "setuptools"}
TOOLCHAIN_DISTS = {"psd-tools", "scipy", "fonttools", "uharfbuzz", "freetype-py", "aggdraw", "cython", "setuptools"}
LOCAL = {"config", "commands"}                  # config.py is per machine (untracked); commands/ is a namespace


def _module(name: str):
    path = REPO / (name.replace(".", "/") + ".py")
    return path if path.is_file() else None


def _imports(path, name: str):
    """(module, line) of every import in the file, at any depth (lazy ones too), and import_module("...") and
    __import__("...") calls."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from ((a.name, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(name.split(".")[:-node.level] + [node.module or ""]).strip(".") if node.level else node.module
            yield base, node.lineno
            yield from ((f"{base}.{a.name}", node.lineno) for a in node.names if _module(f"{base}.{a.name}"))
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("import_module", "__import__")):
            yield node.args[0].value, node.lineno


def bot_closure() -> tuple:
    """({repo module: the import chain that reaches it}, {third-party top: [module:line]}) from main.py."""
    chains, tops, todo = {"main": "main"}, {}, ["main"]
    while todo:
        name = todo.pop()
        for target, line in _imports(_module(name), name):
            top = target.split(".")[0]
            if _module(target):
                if target not in chains:
                    chains[target] = f"{chains[name]} -> {target} ({name}:{line})"
                    todo.append(target)
            elif top not in LOCAL and top not in sys.stdlib_module_names and not _module(top):
                tops.setdefault(top, []).append(f"{chains[name]} ({name}:{line})")
    return chains, tops


def _pins() -> dict:
    lines = [x.strip() for x in (REPO / "requirements-bot.txt").read_text(encoding="utf-8").splitlines()]
    pins = [x for x in lines if x and not x.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+==[^=\s]+", x) for x in pins), "every line one name==version"
    return {re.sub(r"[-_.]+", "-", x.split("==")[0]).lower(): x.split("==")[1] for x in pins}


def test_the_bot_never_reaches_the_card_toolchain():
    chains, tops = bot_closure()
    assert "commands.card_grid" in chains                   # only reached lazily (query_card's grid)
    reached = {top: where for top, where in tops.items() if top in TOOLCHAIN}
    assert not reached, "the bot's closure imports the card toolchain:\n" + "\n".join(
        f"{top}: {where[0]}" for top, where in sorted(reached.items()))
    pins, dists = _pins(), packages_distributions()
    assert not set(pins) & TOOLCHAIN_DISTS and pins["davey"] == "0.1.7"
    for top, where in sorted(tops.items()):
        names = {re.sub(r"[-_.]+", "-", d).lower() for d in dists.get(top, [])}
        assert names & set(pins), f"{top} ({where[0]}): no requirements-bot.txt pin for {sorted(names) or 'it'}"

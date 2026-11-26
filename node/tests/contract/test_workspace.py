"""This workspace's own self-tests: hermetic checks on the code your lineage carries."""

import ast
import importlib
import json
import pathlib

def _root():
    here = pathlib.Path(__file__).absolute()
    for candidate in here.parents:
        if (candidate / "task_agents").is_dir():
            return candidate
    raise AssertionError(f"no workspace root above {here}: none carries task_agents")

ROOT = _root()

def test_carried_folders_import():
    for name in ("meta", "task_agents"):
        if (ROOT / name).is_dir():
            importlib.import_module(name)

def test_task_agents_keep_the_forward_contract():
    if not (ROOT / "task_agents").is_dir():
        return
    for role in sorted(p for p in (ROOT / "task_agents").iterdir() if p.is_dir()):
        assert any(role.iterdir()), f"agent folder {role.name} is empty"
        module = importlib.import_module(f"task_agents.{role.name}")
        forward = getattr(module, "forward", None)
        if forward is not None:
            assert callable(forward), f"task_agents.{role.name}.forward is not callable"

def test_the_expansion_seat_still_starts():
    seat = ROOT / "meta"
    if not seat.is_dir():
        return
    entries = sorted(path for path in seat.glob("entry.*") if path.is_file())
    assert entries, (
        "meta/ carries no entry.*; the expansion seat cannot start, so this"
        " lineage writes no successor and nothing will say why"
    )
    for entry in entries:
        source = entry.read_text(encoding="utf-8", errors="replace")
        if entry.suffix == ".py":
            ast.parse(source, filename=str(entry))
            tree = ast.parse(source)
            defined = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert "main" in defined, f"{entry.name} defines no main(); nothing can start it"
        else:
            assert source.strip(), f"{entry.name} is empty; the expansion seat cannot start"

def test_meta_state_files_stay_parseable():
    roots = [ROOT, *(path for path in (ROOT / "meta",) if path.is_dir())]
    for folder in roots:
        for path in sorted(folder.glob("*_state.json")):
            json.loads(path.read_bytes())

def test_iteration_budget_is_a_usable_number():
    seat = ROOT / "meta"
    if not seat.is_dir():
        return
    sources = sorted(seat.rglob("*.py"))
    if not sources:
        return
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "STEP_LIMIT" for target in node.targets
            ):
                where = path.relative_to(ROOT)
                assert isinstance(node.value, ast.Constant), f"{where}: STEP_LIMIT is not a plain number"
                assert isinstance(node.value.value, int), f"{where}: STEP_LIMIT is not a whole number"
                assert node.value.value > 0, f"{where}: STEP_LIMIT is not positive; no episode could run"
                return
    looked = ", ".join(str(path.relative_to(ROOT)) for path in sources) or "nothing"
    raise AssertionError(
        f"no file under meta/ declares STEP_LIMIT; nothing bounds your episode. Looked in: {looked}"
    )

def test_the_runtime_this_node_carries_is_reachable():
    importlib.import_module("task_agents")
    for name, module in (("shared", "shared.gateway.client"),):
        if (ROOT / name).is_dir():
            importlib.import_module(module)

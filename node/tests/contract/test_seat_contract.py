"""Seat self-tests: every seat this node carries, driven from the descriptor."""

import ast
import importlib
import inspect
import re

import pytest

try:
    from conftest import ROOT
except ImportError:  # outside pytest's rootdir import, as an agent's `python -c` runs it
    from tests.conftest import ROOT

TABLES = ROOT / "tests" / "seats"
PROBE = b"STAGED PROBE BYTES"

class Chat:
    """A programmable stand-in for the model, recording what it was asked."""

    def __init__(self, *replies: str) -> None:
        """Programme this stand-in with the replies it may give, in order."""
        self.replies: list[str] = list(replies)
        self.asked: list[str] = []

    def __call__(self, asked: str) -> str:
        """Record one prompt and answer with the next programmed reply."""
        self.asked.append(asked)
        if not self.replies:
            raise AssertionError(
                f"the seat called chat {len(self.asked)} time(s) and this test"
                " programmed fewer replies; a stand-in does not invent one"
            )
        return self.replies.pop(0)

def _source(role: str) -> str:
    """Return one seat's own source, or "" where the folder carries no module."""
    module = ROOT / "task_agents" / role / "__init__.py"
    return module.read_text(encoding="utf-8", errors="replace") if module.is_file() else ""

def _model_seats(contract) -> list[str]:
    """Return the roles whose whole outward behaviour is one model call."""
    return sorted(
        name
        for name in contract["seeded_agents"]
        if 'inputs.get("chat")' in _source(name)
        and "import subprocess" not in _source(name)
        and _handed(name) == {"chat"}
    )

def _handed(role: str) -> set[str]:
    """Return every capability one seat reaches for in its own source."""
    return set(re.findall(r'inputs\.get\("(\w+)"\)', _source(role)))

def _staged(contract, role: str) -> dict[str, bytes]:
    """Return the staged file map for one role, on the channel it declares."""
    channel = contract["reads"].get(role)
    return {} if channel is None else {channel: PROBE}

def _drive(contract, role: str, *replies: str) -> tuple[dict[str, object], Chat] | None:
    """Drive one seat's forward once, against a stand-in that has no default."""
    seat = importlib.import_module(f"task_agents.{role}")
    chat = Chat(*replies)
    staged = _staged(contract, role)
    try:
        return seat.forward({"files": staged, "seed": 0, "chat": chat}), chat
    except KeyError:
        assert not staged, (
            f"{role} was staged its declared channel {sorted(staged)} and"
            " still refused it: the channel its code reads and the channel the"
            " kernel stages it on have drifted apart"
        )
        return None

def test_every_seeded_agent_imports_and_takes_the_one_call(contract):
    """Every seat the host names imports and exposes a one-argument `forward`."""
    for name in contract["seeded_agents"]:
        importlib.import_module(f"task_agents.{name}")
    for role in contract["roles"]:
        module = importlib.import_module(f"task_agents.{role}")
        forward = getattr(module, "forward", None)
        assert callable(forward), f"role {role} exposes no callable forward; nothing can run it"
        inspect.signature(forward).bind("subject")

def test_a_model_seat_refuses_without_a_gateway_grant(contract):
    """A seat whose whole behaviour is a model call refuses, rather than answering, without a grant."""
    driven = _model_seats(contract)
    for role in driven:
        seat = importlib.import_module(f"task_agents.{role}")
        unreached = Chat()
        for grant in ({}, {"chat": None}):
            try:
                seat.forward({"files": _staged(contract, role), "seed": 0, **grant})
            except RuntimeError:
                continue
            raise AssertionError(f"{role} answered with no gateway grant ({grant})")
        assert unreached.asked == []
    mentions = sorted(name for name in contract["seeded_agents"] if "chat" in _source(name))
    unexplained = [
        name
        for name in mentions
        if name not in driven
        and "import subprocess" not in _source(name)
        and "chat" not in _handed(name)
    ]
    assert not unexplained, (
        f"{unexplained} mention a model and were not driven, and neither"
        " exclusion explains them: this file's own classifier no longer"
        " recognises how your seat is handed its model, so every drive above"
        " passed by skipping it"
    )

def test_a_model_seat_sends_what_it_was_staged_and_more(contract):
    """One model call per seat, carrying the staged bytes and an instruction beyond them."""
    for role in _model_seats(contract):
        driven = _drive(contract, role, '{"Decision": "Accept", "response": "x", "points": 0}')
        if driven is None:
            continue
        result, chat = driven
        assert len(chat.asked) == 1, f"{role} made {len(chat.asked)} model calls, expected one"
        asked = chat.asked[0]
        assert asked.strip(), f"{role} sent an empty prompt"
        staged = _staged(contract, role)
        if staged:
            assert PROBE.decode() in asked, (
                f"{role} did not send what it was staged on its declared channel"
                f" {sorted(staged)}; a seat answering without reading its input"
                " scores on every surface without doing the work"
            )
            assert len(asked) > len(PROBE), (
                f"{role} sent the staged bytes and nothing else, so it asked the"
                " model no question; if your instruction is gone, your edit"
                " removed it -- or matched nothing and you are reading the"
                " previous prompt's number"
            )
        assert isinstance(result, dict) and set(result) >= {"files"}

def test_a_model_seat_answers_in_bytes_and_only_in_bytes(contract):
    """Files and evidence are name-to-bytes mappings, and a seat writes at least one file."""
    for role in _model_seats(contract):
        driven = _drive(contract, role, '{"Decision": "Reject", "response": "x", "points": 7}')
        if driven is None:
            continue
        result, _ = driven
        for half in ("files", "evidence"):
            entries = result.get(half, {})
            assert isinstance(entries, dict), (
                f"{role}: {half} is {type(entries).__name__}, not a mapping of name to bytes"
            )
            for name, body in entries.items():
                assert isinstance(name, str) and name, f"{role}: an unnamed {half} entry"
                assert isinstance(body, bytes), (
                    f"{role}: {half}[{name}] is {type(body).__name__}, not bytes"
                )
        assert result["files"], (
            f"{role} produced no file. Whatever it is judged on, it is judged on"
            " what it wrote, and it wrote nothing"
        )

def test_a_model_seat_is_a_function_of_its_inputs(contract):
    """The same staging and the same reply write the same bytes."""
    for role in _model_seats(contract):
        reply = '{"Decision": "Accept", "response": "x", "points": 3}'
        once, twice = _drive(contract, role, reply), _drive(contract, role, reply)
        if once is None or twice is None:
            continue
        (first, _), (second, _) = once, twice
        assert first["files"] == second["files"], (
            f"{role} wrote different bytes for one staging and one reply; it is"
            " reading something that is not its input"
        )

def test_the_node_ships_a_seat_table(contract):
    """The domain layer ships a reply-shape table under tests/seats naming every role."""
    tables = sorted(TABLES.glob("test_*.py")) if TABLES.is_dir() else []
    assert tables, (
        f"this node carries no seat table at {TABLES.relative_to(ROOT)}/. The"
        " checks in this file are the ones every seat shares; what your own"
        " parser does with a reply it cannot read is not among them. Write a"
        " table of reply shapes there: one row per shape, the score it must"
        " settle, and why"
    )
    named = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in tables)
    unnamed = sorted(role for role in contract["roles"] if role not in named)
    assert not unnamed, f"the seat tables name no rows for {unnamed}"

CLAUSE = 25

def _clauses(rendered: str) -> list[str]:
    """Return the substantial sentences of one rendered prompt."""
    found = []
    for line in rendered.splitlines():
        for fragment in line.replace(". ", ".\n").splitlines():
            if len(fragment.strip()) >= CLAUSE:
                found.append(fragment.strip())
    return found

def test_no_seat_sends_the_same_clause_twice(contract):
    """A prompt edit that left the old clause beside the new one hands the model two policies."""
    for role in _model_seats(contract):
        driven = _drive(contract, role, '{"Decision": "Accept", "response": "x", "points": 0}')
        if driven is None:
            continue
        _, chat = driven
        clauses = _clauses(chat.asked[0])
        twice = sorted({line for line in clauses if clauses.count(line) > 1})
        assert not twice, (
            f"{role} sends {len(twice)} clause(s) twice in one prompt: {twice}."
            " A `str.replace` that included the literal seam SUCCEEDS and leaves"
            " the old clause beside the new one, so the model is given two"
            " policies and follows whichever it likes. Read the rendered prompt,"
            " not the file"
        )

RECOVERY = ("search", "match", "fullmatch", "findall", "loads")

def _late_recovery(source: str) -> list[str]:
    """Return every place a recovery attempt sits after an except that returns."""
    found = []
    for scope in ast.walk(ast.parse(source)):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        returns = [
            handler.lineno
            for node in ast.walk(scope)
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if any(isinstance(inner, ast.Return) for inner in ast.walk(handler))
        ]
        if not returns:
            continue
        earliest = min(returns)
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in RECOVERY
                and node.lineno > earliest
            ):
                found.append(
                    f"{scope.name}: a `{node.func.attr}` at line {node.lineno} sits"
                    f" after an except that returns at line {earliest}"
                )
    return found

def test_no_seat_puts_a_recovery_after_an_except_that_returns(contract):
    """A parse fallback written after a returning except never runs; it is refused here."""
    for name in contract["seeded_agents"]:
        late = _late_recovery(_source(name)) if _source(name) else []
        assert not late, (
            f"{name} carries an unreachable recovery:\n  " + "\n  ".join(late) + "\n"
            "  The handler above it already returned for exactly the replies"
            " this was added to rescue, so it never runs. Try the candidates"
            " BEFORE anything returns, and refuse once they are exhausted."
        )

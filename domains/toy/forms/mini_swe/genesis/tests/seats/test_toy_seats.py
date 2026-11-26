"""This domain's own seat tables: what each seat does with each input shape."""

import json
import pathlib

from task_agents import judge, solver

STRICTNESS_SHAPES = (
    ("0.5", "7", 0.7, False, "0.5 + (7 % 5)/10"),
    ("0.5", "5", 0.5, False, "a residue of zero adds nothing"),
    ("0.0", "4", 0.4, False, "the increment alone"),
    ("0.9", "9", 1.0, False, "the CEILING clamps; 0.9 + 0.4 would be 1.3"),
    ("1.0", "0", 1.0, False, "already at the ceiling"),
    ("0.5", "-3", 0.7, True, "PINS A CHOICE: Python's % is non-negative, so -3 % 5 is 2"),
    ("0.5", " 7 ", 0.7, True, "PINS A CHOICE: int() tolerates surrounding whitespace"),
)

def test_the_judge_holds_its_strictness_law():
    for strictness, answer, expected, pinned, why in STRICTNESS_SHAPES:
        got = judge.verdict(strictness, answer)
        kind = "PINNED CHOICE" if pinned else "LAW"
        assert abs(got - expected) < 1e-9, (
            f"judge.verdict({strictness!r}, {answer!r}) settled {got},"
            f" expected {expected} ({kind}): {why}"
        )
    assert judge.verdict(strictness="0.5", answer="7") == judge.verdict("0.5", "7")
    assert (judge.CEILING, judge.DIVISOR, judge.INCREMENT) == (1.0, 5, 10.0)
    for bad in (("high", "7"), ("0.5", "seven"), ("", "7"), ("0.5", "")):
        try:
            judge.verdict(*bad)
        except ValueError:
            continue
        raise AssertionError(f"the judge settled a verdict on {bad!r}")

def test_the_two_seats_meet_on_one_channel():
    assert solver.ANSWER == judge.PROBE
    produced = solver.forward({"files": {}, "seed": 1, "rule": "strictness-v1"})
    answer = produced["files"][solver.ANSWER]
    assert isinstance(answer, bytes) and answer.strip()
    settled = judge.forward({"files": {judge.PROBE: answer}, "seed": 1, "rule": judge.RULE})
    scored = json.loads(settled["files"][judge.VERDICT])
    strictness = (pathlib.Path(judge.__file__).parent / judge.GENOME).read_text(encoding="utf-8")
    assert scored["score"] == judge.verdict(strictness, answer.decode())
    assert scored["outcome"] == scored["score"]
    try:
        judge.forward({"files": {}, "seed": 1, "rule": judge.RULE})
    except KeyError:
        return
    raise AssertionError("the judge settled a verdict over an empty staging")

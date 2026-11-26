"""What this node's two seats promise, checked the way the host holds them to it.

`CONTRACT.md` beside this file is the inventory: every designed behaviour beside the
test that pins it.

Run from the composed tree (`python -m pytest tests/seats`
inside it): the seats import `shared` and read `.rqgm/contract.json`, and only the composed tree
carries both.
"""

import json
import pathlib
import subprocess

import pytest

from task_agents import reviewer, solver
from task_agents.shared import io_contract

ROOT = next(p for p in pathlib.Path(__file__).absolute().parents if (p / "task_agents").is_dir())
CONTRACT = json.loads((ROOT / ".rqgm" / "contract.json").read_bytes())
POOL = pathlib.Path("/run/rqgm/pool.sock")


class Chat:
    """Answers with the replies it was given, in order, and keeps every prompt it was asked."""

    def __init__(self, *replies: str) -> None:
        self.replies, self.asked = list(replies), []

    def __call__(self, asked: str) -> str:
        self.asked.append(asked)
        return self.replies.pop(0)


def accept(reasoning: str = "r") -> str:
    return json.dumps({"Reasoning": reasoning, "Decision": "Accept"})


WORK = reviewer.staged(problem="P", diff="D").encode()
WORK_FILES = {reviewer.REVIEWED: WORK, reviewer.PATCH: b"D"}
CELL = json.dumps({"instruction": "P", "repository": "o/r", "language": "go", "path": "/app"}).encode()
EPISODE = {"patch": "THE DIFF", "messages": [{"role": "assistant", "content": "done"}], "exit_status": "done", "turns": 2, "tokens": 9}

# Reply shape -> score. The verdict is the one recognised Decision in the block the seat asked
# for; anything else is MALFORMED, which the settlement never mistakes for a reject.
SHAPES = (
    ('{"Reasoning": "r", "Decision": "Accept"}', 1.0),
    ('{"Reasoning": "r", "Decision": "Reject"}', 0.0),
    ('{"Decision": "reject"}', 0.0),
    ('Reviewed.\n{"Decision": "Accept"}\n', 1.0),
    ('```json\n{"Decision": "Accept"}\n```', 1.0),
    ('[{"Decision": "Accept"}]', -1.0),
    ('{"verdict": {"Decision": "Accept"}}', -1.0),
    ('{"Decision": null}', -1.0),
    ('{"Decision": "Merge"}', -1.0),
    ('{"Decision": "Accept"}\n\nOn reflection: {"Decision": "Reject"}', -1.0),
    ("I could Accept this diff, but on balance I Reject it.", -1.0),
    ("", -1.0),
)


def test_the_reviewer_reads_every_reply_shape_the_same_way():
    """A node that changes the parser learns here which replies it stopped or started reading."""
    for reply, expected in SHAPES:
        assert reviewer.verdict(reply=reply) == expected, reply
    assert reviewer.MALFORMED not in reviewer.DECISIONS.values()


def test_the_reviewer_judges_what_the_host_stages_and_answers_in_the_declared_files():
    """The separate raw patch leaves the default prompt on the complete envelope once."""
    chat = Chat(accept())
    answer = reviewer.forward({"files": WORK_FILES, "seed": 0, "chat": chat})
    assert chat.asked == [reviewer.prompt(channel=WORK.decode())]
    assert json.loads(answer["files"][reviewer.VERDICT]) == {"score": 1.0}
    assert answer["evidence"]["reviewed"] == WORK
    with pytest.raises(RuntimeError):
        reviewer.forward({"files": WORK_FILES, "seed": 0, "chat": None})


def test_the_reviewer_refuses_a_staging_that_is_not_the_work():
    """Nothing, a reference alone, or an undeclared name never becomes a verdict."""
    for files in ({}, {reviewer.REFERENCE_SOLUTION: b"gold"}, WORK_FILES | {"review_label": b"accept"}):
        with pytest.raises(KeyError):
            reviewer.channel(files=files)


def test_the_reviewer_sees_the_authors_own_tests_in_the_diff():
    """Any test in the diff is the author's, never the graded one, and the model sees it as the author's intent.

    The graded tests never reach the diff, so a test there is one the author wrote. The instruction
    tells the reviewer to read it as intent, not as the standard, so hiding it would hide intent.
    """
    diff = "diff --git a/x_test.go b/x_test.go\n+func TestX() {}\ndiff --git a/x.go b/x.go\n+func X() {}\n"
    body = reviewer.channel(files={reviewer.REVIEWED: reviewer.staged(problem="P", diff=diff).encode(), reviewer.PATCH: diff.encode()})
    assert "x.go" in body and "x_test.go" in body
    assert body == reviewer.staged(problem="P", diff=diff)
    assert "separate from the true golden tests" in reviewer.INSTRUCTION


def test_an_unreadable_reply_is_asked_again_with_the_same_question():
    """Up to MALFORMED_RETRIES more askings, identical text, and the first readable reply settles."""
    chat = Chat("no verdict here", accept())
    answer = reviewer.forward({"files": WORK_FILES, "seed": 0, "chat": chat})
    assert chat.asked[0] == chat.asked[1] and len(chat.asked) == 2
    assert json.loads(answer["evidence"]["parsed"])["attempts"] == 2
    torn = Chat(*["nothing"] * (reviewer.MALFORMED_RETRIES + 1))
    answer = reviewer.forward({"files": WORK_FILES, "seed": 0, "chat": torn})
    assert json.loads(answer["files"][reviewer.VERDICT]) == {"score": reviewer.MALFORMED}


def test_a_self_review_carries_the_authors_test_output_and_nothing_else_in_its_place():
    """The solver's own test run is the one piece of evidence a self-review adds; a patch or nothing is refused."""
    chat = Chat(accept())
    assert reviewer.self_review(problem="P", diff="D", author_output=b"--- FAIL: TestX", chat=chat) == 1.0
    assert reviewer.AUTHOR_MARK.strip() in chat.asked[0] and "--- FAIL: TestX" in chat.asked[0]
    assert reviewer.INSTRUCTION in chat.asked[0]
    with pytest.raises(ValueError):
        reviewer.self_review(problem="P", diff="D", author_output=b" \n", chat=Chat())
    with pytest.raises(KeyError):
        reviewer.self_review(problem="P", diff="D", author_output=b"diff --git a/t b/t\n@@ -1 +1 @@\n+x", chat=Chat())


@pytest.mark.parametrize("entry", ["review", "self_review"])
def test_co_located_calls_reach_the_edited_forward_and_its_physical_patch(monkeypatch, entry):
    """The composed seat's callable adapters run its edited entry point with readable input paths."""
    seen = []

    def edited(inputs):
        paths = {name: pathlib.Path(path) for name, path in inputs["paths"].items()}
        assert set(paths) == set(inputs["files"]) == set(WORK_FILES)
        assert paths[reviewer.PATCH].read_bytes() == inputs["files"][reviewer.PATCH] == b"D"
        assert paths[reviewer.REVIEWED].read_bytes() == inputs["files"][reviewer.REVIEWED]
        seen.append(paths[reviewer.PATCH])
        return {"files": {reviewer.VERDICT: b'{"score": 0}'}}

    monkeypatch.setattr(reviewer, "forward", edited)
    if entry == "review":
        assert reviewer.review(problem="P", work="D", chat=Chat()) == (0.0, "", [])
    else:
        assert reviewer.self_review(problem="P", diff="D", author_output=b"passed", chat=Chat()) == 0.0
    assert len(seen) == 1 and not seen[0].exists()


def test_the_solver_asks_the_episode_the_statement_and_publishes_the_reviewers_envelope():
    """The cell becomes one prose task; the patch comes back beside the exact bytes the reviewer judges."""
    asked = []
    answer = solver.forward({"files": {"0": CELL}, "seed": 1, "chat": Chat(), "solve": lambda task: asked.append(task) or EPISODE})
    # The preamble is the seat's own line above the benchmark's statement, and the search edits it
    # like any other (5e396b4b0); this pins its place, not its words.
    assert asked == [solver.PREAMBLE.strip() + "\n\nP\n\n### REPOSITORY\no/r\n\n### LANGUAGE\ngo\n\n### WORKING DIRECTORY\n/app"]
    assert answer["files"][solver.ASSESSED].decode() == reviewer.staged(problem="P", diff="THE DIFF")
    assert answer["files"][solver.PATCH] == b"THE DIFF"
    assert json.loads(answer["evidence"]["stop"])["turns"] == 2


def test_the_solver_refuses_what_is_not_a_task_before_any_episode_runs():
    """No grant, no harness, no bytes, no JSON, or a cell missing a field: refused, and the harness never asked."""
    ran = []
    def solve(task):
        ran.append(task)
        return EPISODE
    for inputs, refusal in (
        ({"files": {"0": CELL}, "seed": 1, "chat": None, "solve": solve}, RuntimeError),
        ({"files": {"0": CELL}, "seed": 1, "chat": Chat()}, RuntimeError),
        ({"files": {"0": b""}, "seed": 1, "chat": Chat(), "solve": solve}, KeyError),
        ({"files": {"0": b"not a cell"}, "seed": 1, "chat": Chat(), "solve": solve}, TypeError),
        ({"files": {"0": b'{"instruction": "P"}'}, "seed": 1, "chat": Chat(), "solve": solve}, ValueError),
    ):
        with pytest.raises(refusal):
            solver.forward(inputs)
    assert ran == []


def test_every_seat_answer_is_declared_by_the_arm_that_runs_it():
    """`runtime._forward` refuses an undeclared file or evidence the instant a seat returns; check it here first."""
    driven = (
        ("reviewer", "generate", reviewer.forward, {"files": WORK_FILES, "seed": 0, "chat": Chat(accept())}),
        ("reviewer", "judge", reviewer.forward, {"files": WORK_FILES, "seed": 0, "chat": Chat(accept())}),
        ("solver", "generate", solver.forward, {"files": {"0": CELL}, "seed": 1, "chat": Chat(), "solve": lambda _: EPISODE}),
    )
    for role, mode, seat, inputs in driven:
        arm = CONTRACT["io"][role][mode]
        io_contract.check_inputs(role=role, mode=mode, files=inputs["files"], arm=arm)
        io_contract.check_output(role=role, mode=mode, answer=seat(inputs), arm=arm)


def test_both_reviewer_arms_refuse_the_envelope_without_the_patch_file():
    """A lost patch file refuses before even a rewritten seat can run."""
    for mode in ("generate", "judge"):
        arm = CONTRACT["io"]["reviewer"][mode]
        assert set(arm["reads"]) == set(WORK_FILES)
        with pytest.raises(io_contract.IoRefusalError, match="model.patch"):
            io_contract.check_inputs(
                role="reviewer", mode=mode, files={reviewer.REVIEWED: WORK}, arm=arm,
            )


def test_rqgm_call_samples_lists_this_forms_roles_and_no_label():
    """Inside an expansion the pool answers `samples` with the train rows per role; a label never appears."""
    if not POOL.is_socket():
        pytest.skip(f"no pool at {POOL}: not inside an expansion episode")
    out = subprocess.run(["rqgm-call", "samples"], check=True, capture_output=True, text=True, timeout=120).stdout
    listed = json.loads(out.splitlines()[0])
    assert listed["ok"] is True
    assert set(CONTRACT["roles"]) <= set(map(str, listed.get("roles", listed.get("samples", {}))))
    assert "review_label" not in out

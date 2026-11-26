"""What a node relies on from this form's two seats, checked without a model.

`tests/contract/` holds what every seat everywhere shares; this holds what is true of the imo
grader and writer: the reply shapes the grader reads, the layout it marks, what the writer files,
how the two consult each other, and that every evidence key a seat states is one its arm declares.
`CONTRACT.md` beside this file lists every designed behaviour of the seats and the test here that
pins it.
"""

import json
import pathlib

from task_agents import grader, writer
from task_agents.shared import io_contract


class Chat:
    """A stand-in for the model: answers from a script, records what it was asked, never invents."""

    def __init__(self, *replies: str) -> None:
        self.replies, self.asked = list(replies), []

    def __call__(self, asked: str) -> str:
        self.asked.append(asked)
        assert self.replies, f"the seat called chat {len(self.asked)} time(s), more than this test programmed"
        return self.replies.pop(0)


BLOCK = '```json\n{"points": %d}\n```'
LATEX = r"The claim \(p \mid N-1\) forces \(p^e \mid F_N\) only when \(\gcd(p, N) = 1\), and \pmod{3} the case \frac{a}{b} is unproved."

# (reply, mark, why). A malformed reply is -1, which is off the scale on purpose.
MARK_SHAPES = (
    ('{"points": 5}', 5, "the block alone"),
    (BLOCK % 7, 7, "the block fenced, as the instruction asks"),
    ("Complete and rigorous.\n\n" + BLOCK % 7, 7, "reasoning first, then the fenced block"),
    (LATEX + "\n\n" + BLOCK % 3, 3, "LaTeX braces and backslashes in the reasoning do not reach the block"),
    ("Not a proof. {\"points\": 0}", 0, "an unfenced block after prose is read through the outermost braces"),
    (BLOCK % 3 + "\n\nOn reflection:\n\n" + BLOCK % 6, -1, "two different marks is no mark"),
    ("I award\n\n" + BLOCK % 4 + "\n\n" + BLOCK % 4, 4, "the same mark stated twice is one mark"),
    (BLOCK % 9, -1, "off the scale, never clamped"),
    (BLOCK % -1, -1, "below the scale"),
    ('```json\n{"points": "6"}\n```', -1, "a string is not an integer mark"),
    ('```json\n{"points": 6.0}\n```', -1, "a float is not an integer mark"),
    ('```json\n{"points": true}\n```', -1, "a boolean is not an integer mark"),
    ('```json\n{"mark": 6}\n```', -1, "the wrong field"),
    ("I would give this a 6.", -1, "no block: a mark not committed to"),
    ("", -1, "nothing came back"),
)


def test_the_grader_reads_every_reply_shape():
    for reply, expected, why in MARK_SHAPES:
        assert grader.mark(reply=reply) == expected, f"grader.mark({reply!r}) -> {grader.mark(reply=reply)}, expected {expected}: {why}"
    assert (grader.LOW, grader.HIGH) == (0, 7)
    assert not grader.LOW <= grader.MALFORMED <= grader.HIGH, "malformed must never look like a mark"


def test_the_grader_prompt_puts_the_work_first_and_asks_for_the_block_once():
    rendered = grader.prompt(work="THE WORK")
    assert rendered.startswith("THE WORK\n\n")
    assert rendered.count('{"points": 0}') == 1
    assert rendered.count("0 to 7") >= 1


def test_the_grader_marks_a_judged_proof_in_the_corpus_layout():
    chat = Chat("Solid work. " + BLOCK % 6)
    result = grader.forward({"files": {grader.GUIDANCE: b"### Problem\n\nP", grader.MARKING: b"the proof"}, "seed": 4, "chat": chat})
    assert chat.asked == [grader.prompt(work=grader.judged(guidance=b"### Problem\n\nP", attempt=b"the proof"))]
    assert chat.asked[0].index(grader.RESPONSE_HEADING) < chat.asked[0].index("the proof")
    assert json.loads(result["files"][grader.GRADE]) == {"score": 6}
    assert json.loads(result["evidence"]["parsed"]) == {"field": "points", "malformed": False, "score": 6}
    assert result["evidence"]["reply"] == ("Solid work. " + BLOCK % 6).encode()


def test_the_grader_marks_a_corpus_row_as_staged():
    chat = Chat(BLOCK % 1)
    result = grader.forward({"files": {grader.MARKING: b"### Problem\n\nP\n\n### Response\n\nR"}, "seed": 4, "chat": chat})
    assert chat.asked == [grader.prompt(work="### Problem\n\nP\n\n### Response\n\nR")]
    assert json.loads(result["files"][grader.GRADE]) == {"score": 1}


def test_the_grader_records_an_unreadable_reply_as_malformed_not_zero():
    result = grader.forward({"files": {grader.MARKING: b"work"}, "seed": 4, "chat": Chat("I cannot decide.")})
    assert json.loads(result["files"][grader.GRADE]) == {"score": grader.MALFORMED}
    assert json.loads(result["evidence"]["parsed"])["malformed"] is True


def test_the_grader_refuses_no_work_and_no_grant():
    for inputs, refusal in (({"files": {}, "seed": 4, "chat": Chat()}, KeyError), ({"files": {grader.MARKING: b"w"}, "seed": 4, "chat": None}, RuntimeError)):
        try:
            grader.forward(inputs)
        except refusal:
            continue
        raise AssertionError(f"the grader ran on {inputs!r} instead of raising {refusal.__name__}")


def test_the_grader_reviews_a_draft_and_returns_mark_and_reasoning():
    reply = "Case n=2 is asserted, not proved.\n\n" + BLOCK % 3
    chat = Chat(reply)
    mark, reasoning, trace = grader.review(problem="Prove P.", work="Suppose not.", chat=chat)
    assert (mark, reasoning) == (3.0, reply)
    assert trace == [{"prompt": chat.asked[0], "reply": reply}]
    asked = chat.asked[0]
    assert asked.index("Prove P.") < asked.index(grader.DRAFTED) < asked.index("Suppose not.") < asked.index(grader.INSTRUCTION)


def test_the_writer_files_the_reply_whole_and_sends_the_problem_first():
    chat = Chat("Suppose not. Then $x=1$. $\\blacksquare$")
    result = writer.forward({"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": chat})
    assert chat.asked == [writer.prompt(problem="P")]
    assert chat.asked[0].startswith("P\n\n")
    assert result["files"][writer.ATTEMPT] == b"Suppose not. Then $x=1$. $\\blacksquare$"
    assert result["evidence"]["staged"] == b"P"
    assert result["evidence"]["review"].startswith(b"no review")


def test_the_writer_asks_the_grader_once_and_files_the_revision():
    chat = Chat("first draft", "revised proof")
    seen = []

    trace = [{"prompt": "the marking text", "reply": "Case n=2 is asserted, not proved."}]

    def seat(*, problem, work, chat):
        seen.append((problem, work))
        return 3.0, "Case n=2 is asserted, not proved.", trace

    result = writer.forward({"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": chat, "review": seat})
    assert seen == [("P", "first draft")]
    assert chat.asked[1] == writer.revision(problem="P", proof="first draft", mark=3.0, reasoning="Case n=2 is asserted, not proved.", trace=trace)
    for carried in ("P", "first draft", "3.0", "Case n=2 is asserted, not proved.", "the marking text", writer.REVISE):
        assert carried in chat.asked[1], f"the revision prompt carries no {carried!r}"
    assert result["files"][writer.ATTEMPT] == b"revised proof"
    assert json.loads(result["evidence"]["review"]) == {"score": 3.0, "review": "Case n=2 is asserted, not proved.", "dialogue": trace}


def test_a_broken_reviewer_costs_the_revision_and_never_the_proof():
    def broken(*, problem, work, chat):
        raise RuntimeError("seat down")

    chat = Chat("only draft")
    result = writer.forward({"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": chat, "review": broken})
    assert result["files"][writer.ATTEMPT] == b"only draft"
    assert len(chat.asked) == 1


def test_zero_rounds_is_the_one_call_seat(monkeypatch):
    monkeypatch.setattr(writer, "REVIEW_ROUNDS", 0)
    chat = Chat("only draft")
    writer.forward({"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": chat, "review": lambda **_: (7.0, "fine", [])})
    assert len(chat.asked) == 1


def test_the_default_is_one_round_and_is_stated_on_the_module():
    assert writer.REVIEW_ROUNDS == 1


def test_the_writer_refuses_without_a_grant():
    try:
        writer.forward({"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": None})
    except RuntimeError:
        return
    raise AssertionError("the writer wrote without a grant")


# A materialized node carries `.rqgm/contract.json`; a repository checkout carries the form's
# `form.json` above the genesis tree instead. Same fact, two carriers.
ROOT = next(parent for parent in pathlib.Path(__file__).absolute().parents if (parent / "task_agents").is_dir())
NODE_CONTRACT = ROOT / ".rqgm" / "contract.json"
FORM = ROOT.parent / "form.json"


def declared_arm(*, role: str, mode: str) -> dict:
    if NODE_CONTRACT.is_file():
        arm = json.loads(NODE_CONTRACT.read_text(encoding="utf-8"))["io"][role][mode]
        return {"writes": sorted(arm["writes"]), "evidence": sorted(arm["evidence"])}
    episode = json.loads(FORM.read_text(encoding="utf-8"))["roles"][role]["episode"]
    return {"writes": sorted(episode["outputs"]), "evidence": sorted(episode["evidence"])}


DRIVEN = (
    ("writer", "generate", writer.forward, lambda: {"files": {writer.PROBLEM: b"P"}, "seed": 5, "chat": Chat("the proof")}),
    ("grader", "generate", grader.forward, lambda: {"files": {grader.MARKING: b"the proof"}, "seed": 4, "chat": Chat(BLOCK % 6)}),
    ("grader", "judge", grader.forward, lambda: {"files": {grader.GUIDANCE: b"### Problem\n\nP", grader.MARKING: b"the proof"}, "seed": 4, "chat": Chat(BLOCK % 6)}),
)


def test_every_evidence_a_seat_states_is_declared_by_its_arm():
    """The check the scored runtime runs the instant a seat returns; a stray key fails every evaluation."""
    for role, mode, seat, staged in DRIVEN:
        io_contract.check_output(role=role, mode=mode, answer=seat(staged()), arm=declared_arm(role=role, mode=mode))

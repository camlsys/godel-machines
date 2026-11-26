"""The paper seats' contract: what each seat sends, what it writes back, and how it reads a reply.

Run from the node root: `python -m pytest tests/seats`. Each test's first line says why a node
relies on it; `CONTRACT.md` beside this file maps every designed behaviour of the domain to the test
that pins it, here or on the host.
"""

import json
import pathlib

from task_agents import reviewer, writer
from task_agents.shared import io_contract

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / ".rqgm" / "contract.json").read_bytes())

PREMISE = b"Title: A Paper\n\nAbstract: It exists."
PAPER = b"# A Paper\n\nIts whole body."
REVIEW = 'THOUGHT:\nfine.\n\nREVIEW JSON:\n```json\n{"Summary": "s", "Decision": "Accept"}\n```'


class Chat:
    """A stand-in model: answers with the programmed replies in order and never invents one."""

    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def __call__(self, asked):
        self.asked.append(asked)
        assert self.replies, f"the seat made call {len(self.asked)} with {len(self.asked) - 1} replies programmed"
        return self.replies.pop(0)


def test_the_writer_sends_the_premise_first_and_its_instruction_once():
    """The model must see the staged title and abstract, then what to do with them, in one call."""
    chat = Chat("the draft")
    writer.forward({"files": {"0": PREMISE}, "seed": 0, "chat": chat})
    assert chat.asked == [writer.prompt(premise=PREMISE.decode())]
    assert chat.asked[0].startswith(PREMISE.decode())
    assert chat.asked[0].count(writer.INSTRUCTION) == 1


def test_the_writer_writes_the_reply_whole_as_the_draft():
    """The reply is the paper: nothing is parsed out of it, so a fence or LaTeX cannot lose the draft."""
    for reply in ("# A Paper\n\nWith \\(x\\) inside.", '```json\n{"response": "x"}\n```', ""):
        result = writer.forward({"files": {"0": PREMISE}, "seed": 0, "chat": Chat(reply)})
        assert result["files"] == {writer.DRAFT: reply.encode()}
        assert result["evidence"] == {"reply": reply.encode(), "staged": PREMISE, "review": b"no review: this node seats no reviewer, or it did not answer"}


def test_the_writer_refuses_an_empty_premise_before_spending_the_call():
    """A blank staging must not cost a model call and a review of a paper written from nothing."""
    for files in ({}, {"0": b""}, {"0": b"", "1": b""}, {"0": b" \n\t "}):
        chat = Chat()
        try:
            writer.forward({"files": files, "seed": 0, "chat": chat})
        except ValueError as refusal:
            assert "nothing to draft from" in str(refusal)
        else:
            raise AssertionError(f"the writer drafted from {files!r}")
        assert chat.asked == []


def test_the_reviewer_sends_its_instruction_first_and_the_whole_paper_once():
    """The model sees the review form naming the field the seat parses, then the whole paper.

    The instruction leads, as the upstream baseline composes it: a reviewer that moves the form
    behind the paper returns far more unparseable verdicts.
    """
    chat = Chat(REVIEW)
    reviewer.forward({"files": {reviewer.PAPER: PAPER}, "seed": 0, "chat": chat})
    assert chat.asked == [reviewer.prompt(paper=PAPER.decode())]
    assert chat.asked[0].startswith(reviewer.INSTRUCTION)
    assert chat.asked[0].count(PAPER.decode()) == 1
    assert chat.asked[0].count(reviewer.INSTRUCTION) == 1
    assert f'"{reviewer.FIELD}"' in reviewer.INSTRUCTION


def test_the_reviewer_writes_a_numeric_verdict_and_the_three_evidence_files():
    """`verdict.txt` is `{"score": x}`; the reply, the paper read and the parse are the evidence."""
    result = reviewer.forward({"files": {reviewer.PAPER: PAPER}, "seed": 0, "chat": Chat(REVIEW)})
    assert json.loads(result["files"][reviewer.VERDICT]) == {"score": 1.0}
    assert result["evidence"]["reply"] == REVIEW.encode()
    assert result["evidence"]["reviewed"] == PAPER
    assert json.loads(result["evidence"]["parsed"]) == {"field": "Decision", "score": 1.0, "malformed": False}


VERDICTS = (
    # one decision, bare, fenced, or behind prose: read
    ('{"Decision": "Accept"}', 1.0),
    ('{"Decision": "reject"}', 0.0),
    ('```json\n{"Decision": "Accept"}\n```', 1.0),
    (REVIEW.replace("Accept", "Reject"), 0.0),
    ("Here is my review.\n\n" + '{"Summary": "s", "Decision": "Accept"}', 1.0),
    # no single recognised decision: -1.0, a wrong answer on accept and reject rows alike
    ("", -1.0),
    ('{"Summary": "s"}', -1.0),
    ('{"Decision": "Borderline"}', -1.0),
    ('{"Decision": null}', -1.0),
    ('{"Decision": "Accept"}\n\nOn reflection: {"Decision": "Reject"}', -1.0),
    ("I would Accept this paper.", -1.0),
    ('[{"Decision": "Accept"}]', -1.0),
    ('{"review": {"Decision": "Accept"}}', -1.0),
    ('{"Decision": Accept}', -1.0),
)


def test_the_reviewer_reads_one_decision_out_of_each_reply_shape():
    """A node editing the prompt must know which reply shapes score and which are worth -1.0."""
    for reply, expected in VERDICTS:
        assert reviewer.verdict(reply=reply) == expected, f"{reply!r} settled {reviewer.verdict(reply=reply)}"
    assert reviewer.MALFORMED not in reviewer.DECISIONS.values()


def test_the_writer_draft_is_the_channel_the_reviewer_reads():
    """The reviewer can judge the writer's draft only because both seats name one channel."""
    assert writer.DRAFT == reviewer.PAPER == CONTRACT["reads"]["reviewer"]


def test_self_review_is_the_reviewer_reading_a_draft_in_progress():
    """The meta agent's self-review asks this seat the question `forward` asks, over the work so far."""
    chat = Chat(REVIEW)
    score = reviewer.self_review(problem="Improve the writer.", diff="--- a\n+++ b", author_output=b"tests pass", chat=chat)
    assert score == 1.0
    sent = chat.asked[0]
    # The seat composes its instruction first and the paper once (CONTRACT.md), and the self-review
    # is that seat over the draft so far: problem, the drafted marker, the diff, the author's output.
    order = [sent.index(part) for part in (reviewer.INSTRUCTION, "Improve the writer.", reviewer.DRAFTED, "--- a", "tests pass")]
    assert order == sorted(order)


def test_each_seat_answers_within_the_arm_its_form_declares():
    """The runtime refuses an answer naming an undeclared file or evidence, so every scored row would die there."""
    for role, mode, seat, files in (
        ("writer", "generate", writer, {"0": PREMISE}),
        ("reviewer", "generate", reviewer, {reviewer.PAPER: PAPER}),
        ("reviewer", "judge", reviewer, {reviewer.PAPER: PAPER}),
    ):
        answer = seat.forward({"files": files, "seed": 0, "chat": Chat(REVIEW)})
        io_contract.check_output(role=role, mode=mode, answer=answer, arm=CONTRACT["io"][role][mode])


def test_the_writer_is_staged_the_blind_channel_and_the_reviewer_the_body():
    """The writer reads `input` (title and abstract) by position; the reviewer reads `paper_text` by name; neither reads `decision`."""
    assert CONTRACT["reads"] == {"writer": "input", "reviewer": "paper_text"}
    assert CONTRACT["io"]["writer"]["generate"]["positional"] is True
    assert CONTRACT["io"]["writer"]["generate"]["reads"] == []
    assert CONTRACT["io"]["reviewer"]["generate"] == {**CONTRACT["io"]["reviewer"]["generate"], "positional": False, "reads": ["paper_text"]}
    assert "decision" not in {*CONTRACT["reads"].values(), *(name for arm in CONTRACT["io"].values() for vocabulary in arm.values() for name in vocabulary["reads"])}


def test_the_reviewer_has_one_vocabulary_on_both_of_its_arms():
    """Generate (a real paper) and judge (the writer's draft) stage the same channel and owe the same files, so the seat cannot tell which arm it is on."""
    assert set(CONTRACT["io"]["reviewer"]) == {"generate", "judge"}
    assert CONTRACT["io"]["reviewer"]["generate"] == CONTRACT["io"]["reviewer"]["judge"]
    assert set(CONTRACT["io"]["writer"]) == {"generate"}


def test_the_rows_a_node_may_study_are_the_train_halves_and_never_the_measured_ones():
    """Train rows (50 + 50 reviewer, 100 writer) are staged for the node; validation rows (the same counts) decide scores and are never staged; heldout is named nowhere."""
    cells = CONTRACT["dataset_to_rows"]
    assert {cell: rows["practice"] for cell, rows in cells.items()} == {
        "reviewer/review_positive": {"set": "train_positive", "rows": 50},
        "reviewer/review_negative": {"set": "train_negative", "rows": 50},
        "writer/write": {"set": "write_train", "rows": 100},
    }
    assert {cell: rows["measured"] for cell, rows in cells.items()} == {
        "reviewer/review_positive": {"set": "validation_positive", "rows": 50},
        "reviewer/review_negative": {"set": "validation_negative", "rows": 50},
        "writer/write": {"set": "write_validation", "rows": 100},
    }
    assert "heldout" not in json.dumps(CONTRACT)

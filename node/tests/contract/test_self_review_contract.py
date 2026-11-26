"""Self-review: every node answers the same question, in the form the question takes there."""

import importlib.util
import inspect

import pytest

JUDGES, SCORER, NODE_SCORED = "judges", "scorer", "node"
FLAG = "--self-review"


@pytest.fixture
def carried(root):
    """This node's self-review module, or None where no Python harness could serve one."""
    review = _review_module(root)
    if review.is_file():
        spec = importlib.util.spec_from_file_location("carried_self_review", review)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    meta = root / "meta"
    served = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (sorted(meta.rglob("*")) if meta.is_dir() else ())
        if path.is_file() and path.suffix in {".py", ".js", ".mjs", ".txt", ".json"}
    )
    assert FLAG not in served, f"this node advertises {FLAG!r} and carries no self_review.py"
    return None


def _review_module(root):
    """Where this tree carries its self-review module: the shared layer, or meta/ in an older tree."""
    shared = root / "shared" / "self_review.py"
    return shared if shared.is_file() else root / "meta" / "self_review.py"


def _declared(contract):
    """The judging slots this tree states are scored by a module it carries."""
    stated = contract.get(JUDGES)
    if not isinstance(stated, dict):
        return []
    return sorted(slot for slot, spec in stated.items() if isinstance(spec, dict) and spec.get(SCORER) == NODE_SCORED)


def test_a_python_harness_carries_the_review_module(root):
    """A meta/entry.py harness can reach a self_review.py; the shared layer ships it for every tree."""
    pythonic = (root / "meta" / "entry.py").is_file()
    if pythonic:
        assert _review_module(root).is_file()


def test_the_module_answers_this_tree_rather_than_reading_a_name(carried, root, contract):
    """The seat it resolves is one the descriptor declares node-scored, or none when none is."""
    if carried is None:
        return
    seated = carried.review_seat(workspace=root)
    declared = _declared(contract)
    if not declared:
        assert seated is None
        return
    assert seated is not None and seated.split(".")[-1] in declared


def test_a_tree_that_cannot_consult_says_which_silence_it_is(carried, root, contract):
    """Not reviewing always comes with the reason, and the reason names the entry only where a seat exists."""
    if carried is None:
        return
    refused = carried.why_not_reviewing(workspace=root)
    if refused is None:
        assert carried.reviewing(workspace=root) is True
        return
    assert carried.reviewing(workspace=root) is False
    assert (carried.REVIEW_ENTRY in refused) == bool(_declared(contract))


def test_the_advertised_command_names_this_interpreter(carried):
    """The command a solver types ends in the node's own tail and carries the flag."""
    if carried is None:
        return
    advertised = carried.invocation(python=carried.episode_python())
    assert advertised.endswith(carried.INVOCATION_TAIL) and carried.SELF_REVIEW_FLAG in advertised


def test_the_entry_point_takes_the_evidence_it_judges(carried, root):
    """The seat's entry takes the diff and the grant; review takes the episode's own capture."""
    if carried is None or not carried.reviewing(workspace=root):
        return
    seat, _ = carried.consultable(workspace=root)
    taken = set(inspect.signature(getattr(seat, carried.REVIEW_ENTRY)).parameters)
    assert {"diff", "chat"} <= taken
    handed = inspect.signature(carried.feedback).parameters
    assert {"capture", "state", "chat"} <= set(handed)
    assert handed["chat"].default is inspect.Parameter.empty


def test_a_shell_working_beside_the_tree_is_told_where_the_tree_is(carried, root, contract, tmp_path, monkeypatch):
    """A checkout that is not under the tree finds the contract through the episode's word, not by walking up.

    `contract` is taken so a bare genesis view, which carries no `.rqgm/contract.json` for the
    episode's word to point at, skips rather than fails.
    """
    if carried is None:
        return
    beside = tmp_path / "checkout"
    beside.mkdir()
    monkeypatch.delenv(carried.TREE_VAR, raising=False)
    assert carried.locate(start=beside) == beside.resolve()
    assert carried.locate(start=beside, environ={carried.TREE_VAR: str(root)}) == root.resolve()
    monkeypatch.setenv(carried.TREE_VAR, str(root))
    assert carried.locate(start=beside) == root.resolve()
    assert carried.locate(start=beside, environ={carried.TREE_VAR: str(tmp_path / "absent")}) == beside.resolve()


def test_the_diff_is_read_where_the_work_is_and_the_seat_where_the_tree_is(carried, root, tmp_path):
    """A repository episode edits its checkout beside the tree; the review diffs that, not the tree."""
    if carried is None:
        return
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert carried.worked_in(workspace=root, environ={}) == root
    assert carried.worked_in(workspace=root, environ={carried.CHECKOUT_VAR: str(checkout)}) == checkout.resolve()
    assert carried.worked_in(workspace=root, environ={carried.CHECKOUT_VAR: str(tmp_path / "absent")}) == root

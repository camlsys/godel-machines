"""The host's contract names this node's seats, and every seat it names is here and callable."""

import importlib
import inspect


def test_contract_descriptor_is_well_formed(contract):
    """Roles and slots are sorted, non-empty name lists: the shape every other test reads."""
    for key in ("roles", "slots"):
        names = contract[key]
        assert names and all(isinstance(name, str) and name for name in names)
        assert names == sorted(names)


def test_seeded_agent_folders_survive(root, contract):
    """A seat the host seeded still has its folder; a deleted seat cannot be dispatched."""
    for name in contract["seeded_agents"]:
        folder = root / "task_agents" / name
        assert folder.is_dir() and any(folder.iterdir()), f"seeded agent {name} was deleted"


def test_seeded_agents_keep_the_forward_contract(contract):
    """Every seeded seat imports and its `forward` takes the one argument the runtime hands it."""
    for name in contract["seeded_agents"]:
        module = importlib.import_module(f"task_agents.{name}")
        forward = getattr(module, "forward", None)
        if forward is not None:
            assert callable(forward), f"task_agents.{name}.forward is not callable"
            inspect.signature(forward).bind("subject")

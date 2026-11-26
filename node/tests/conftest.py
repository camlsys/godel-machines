# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""What every node test shares: the workspace root, the host's contract, a fake gateway grant."""

import json
import pathlib

import pytest

ROOT = next(
    parent
    for parent in pathlib.Path(__file__).absolute().parents
    if (parent / "task_agents").is_dir()
)

GRANT = {
    "OPENAI_BASE_URL": "http://gateway.invalid/v1",
    "OPENAI_API_KEY": "test-key",
    "RQGM_AGENT_MODEL": "test-model",
    "RQGM_AGENT_TOKEN_ALLOTMENT": "1000",
    "RQGM_AGENT_SURFACE": "chat",
    "RQGM_AGENT_BUDGET_EXPOSE": "tokens",
    "RQGM_AGENT_BUDGET_NUDGE": "0.5",
    "RQGM_AGENT_MAX_SEQUENCE_TOKENS": "unbounded",
    "RQGM_AGENT_MAX_OUTPUT_TOKENS": "unbounded",
    "RQGM_AGENT_TAG": "test-tag",
    "RQGM_AGENT_CALL_SECONDS": "1",
}


@pytest.fixture(scope="session")
def root() -> pathlib.Path:
    """The workspace root: the directory carrying task_agents/."""
    return ROOT


@pytest.fixture(scope="session")
def contract() -> dict:
    """The host's .rqgm/contract.json; a bare genesis view carries none, so those tests skip."""
    path = ROOT / ".rqgm" / "contract.json"
    if not path.is_file():
        pytest.skip(".rqgm/contract.json is absent: the host writes it when it materializes a node")
    return json.loads(path.read_bytes())


@pytest.fixture
def grant(monkeypatch) -> dict:
    """A complete episode environment, pointing at a gateway nothing answers on."""
    for key, value in GRANT.items():
        monkeypatch.setenv(key, value)
    return dict(GRANT)


@pytest.fixture
def node(tmp_path, monkeypatch) -> pathlib.Path:
    """An empty node at depth 1 with a contract for one seat, `probe`, entered as the cwd."""
    (tmp_path / ".rqgm").mkdir()
    (tmp_path / ".rqgm" / "contract.json").write_text(
        json.dumps(
            {
                "depth": 1,
                "roles": ["probe"],
                "slots": ["probe"],
                "seeded_agents": [],
                "reads": {"probe": "paper"},
                "judges": {"probe": {"scorer": "node", "over": ["probe"]}},
                "io": {
                    "probe": {
                        "generate": {
                            "reads": [],
                            "positional": True,
                            "writes": ["answer.txt"],
                            "evidence": ["log.txt"],
                        },
                        "judge": {
                            "reads": ["paper"],
                            "positional": False,
                            "writes": ["verdict.json"],
                            "evidence": ["log.txt"],
                        },
                    },
                    "fixed": {"host-instrument": {"reads": [], "positional": False, "writes": [], "evidence": []}},
                },
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path

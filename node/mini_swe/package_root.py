# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Stand in for upstream's package root without its startup side effects."""

from typing import Any, Protocol

__version__ = "2.4.6"


class Model(Protocol):
    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs: object) -> dict: ...
    def format_message(self, **kwargs: object) -> dict: ...
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]: ...
    def get_template_vars(self, **kwargs: object) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...


class Environment(Protocol):
    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...
    def get_template_vars(self, **kwargs: object) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...


class Agent(Protocol):
    config: Any

    def run(self, task: str, **kwargs: object) -> dict: ...


__all__ = ["Agent", "Environment", "Model", "__version__"]

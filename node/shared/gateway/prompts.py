# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Load fixed harness prompts and apply a form's optional override."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

APPEND: Final = "+++ append"
PREPEND: Final = "+++ prepend"
REPLACE: Final = "+++ replace"

_PROMPTS: Final[dict[str, str]] = json.loads(
    Path(__file__).with_name("prompts.json").read_text(encoding="utf-8"),
)


def shared_meta() -> str:
    """Return the meta-agent prompt body every harness shares."""
    return carried(name="meta_prompt.md").strip()


def carried(*, name: str) -> str:
    """Return one fixed prompt, refusing unknown names."""
    try:
        return _PROMPTS[name]
    except KeyError:
        raise KeyError(f"no prompt named {name!r}; this module holds {sorted(_PROMPTS)}") from None


def filled(*, name: str, fields: Mapping[str, str]) -> str:
    """Return a fixed prompt with each requested field substituted once."""
    body = carried(name=name)
    for key, value in fields.items():
        marker = "{" + key + "}"
        if marker not in body:
            raise KeyError(f"{name}: nothing to substitute for {marker}")
        body = body.replace(marker, value)
    return body


def resolve(*, default: str, carried: Path | None) -> str:
    """Return the prompt one form actually runs under."""
    if carried is None or not carried.is_file():
        return default
    text = carried.read_text(encoding="utf-8")
    marker, _, rest = text.partition("\n")
    mode = marker.strip()
    if mode == APPEND:
        return f"{default}\n\n{rest.strip()}"
    if mode == PREPEND:
        return f"{rest.strip()}\n\n{default}"
    if mode == REPLACE:
        return rest.strip()
    return text.strip()

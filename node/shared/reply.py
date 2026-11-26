# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""Read the JSON block a seat asked for out of a model reply."""

from __future__ import annotations

import json
import re

FENCED = re.compile(r"```[A-Za-z0-9_+-]*\r?\n(.*?)```", re.DOTALL)
_UNPARSED = object()


def _parsed(*, text: str) -> object:
    """Return the JSON value `text` holds, or `_UNPARSED`; `null` is a value and not a failure."""
    try:
        return json.loads(text)
    except ValueError:
        return _UNPARSED


def _blocks(*, text: str) -> list[str]:
    """Keep outer values whole, including arrays and strings whose contents are not replies."""
    blocks: list[str] = []
    stack: list[str] = []
    opened = 0
    quoted = escaped = False
    for index, char in enumerate(text):
        if not stack and not quoted:
            if char not in '[{"]}':
                continue
            opened = index
        if quoted:
            if char == '"' and not escaped:
                quoted = False
            escaped = char == "\\" and not escaped
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}" and (not stack or stack.pop() != char):
            return []
        if not stack and not quoted:
            blocks.append(text[opened : index + 1])
    return [] if stack or quoted else blocks


def candidates(*, reply: str, unwrap: bool = True) -> list[str]:
    """Return the substrings of one reply that may be the block, the whole reply first."""
    if _parsed(text=reply) is not _UNPARSED or not unwrap:
        return [reply]
    found = [reply]
    start = 0
    for fence in FENCED.finditer(reply):
        found.extend(_blocks(text=reply[start : fence.start()]))
        found.append(fence.group(1))
        start = fence.end()
    found.extend(_blocks(text=reply[start:]))
    return found


def field(*, reply: str, name: str, unwrap: bool = True) -> list[object]:
    """Return the value of `name` in every candidate block that is an object carrying it."""
    found: list[object] = []
    for candidate in candidates(reply=reply, unwrap=unwrap):
        parsed = _parsed(text=candidate)
        if isinstance(parsed, dict) and name in parsed:
            found.append(parsed[name])
    return found

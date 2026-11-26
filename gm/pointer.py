# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Address large execution artifacts without embedding them in event payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gm.ids import ArtifactOid

if TYPE_CHECKING:
    from collections.abc import Callable
    from gm.domain.results import ArtifactRef

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_THRESHOLD = 1 << 20
_POINTER = re.compile(
    rb"version https://git-lfs\\.github\\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n"
)


class PointerError(Exception):
    """Refuse an artifact pointer whose stored bytes do not match its address."""


@dataclass(frozen=True, slots=True)
class Pointer:
    oid: ArtifactOid
    size: int


def pointer_bytes(*, oid: ArtifactOid, size: int) -> bytes:
    return f"version {POINTER_VERSION}\noid sha256:{oid}\nsize {size}\n".encode()


def read_pointer(*, data: bytes) -> Pointer | None:
    found = _POINTER.fullmatch(data)
    return None if found is None else Pointer(ArtifactOid(found[1].decode()), int(found[2]))


def enters_as_pointer(*, data: bytes) -> bool:
    return len(data) > POINTER_THRESHOLD


def pointerized(
    *, data: bytes, route: Callable[[bytes], ArtifactRef], fetch: Callable[[ArtifactOid], bytes]
) -> bytes:
    """Route a large artifact once and return its small verified address."""
    if not enters_as_pointer(data=data):
        return data
    oid = ArtifactOid.of_bytes(data)
    route(data)
    _verify(oid=oid, size=len(data), stored=fetch(oid))
    return pointer_bytes(oid=oid, size=len(data))


def materialized(*, data: bytes, fetch: Callable[[ArtifactOid], bytes]) -> bytes:
    """Resolve an artifact pointer; ordinary bytes pass through unchanged."""
    pointer = read_pointer(data=data)
    if pointer is None:
        return data
    stored = fetch(pointer.oid)
    _verify(oid=pointer.oid, size=pointer.size, stored=stored)
    return stored


def _verify(*, oid: ArtifactOid, size: int, stored: bytes) -> None:
    if len(stored) != size or ArtifactOid.of_bytes(stored) != oid:
        raise PointerError(
            f"artifact {oid} promises {size} bytes but stored bytes have length {len(stored)}"
            f" and digest {ArtifactOid.of_bytes(stored)}"
        )

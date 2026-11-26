# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Define shared validated values and canonical byte rules."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, GetCoreSchemaHandler, PlainSerializer
from pydantic_core import core_schema

if TYPE_CHECKING:
    from gm.spend import SpendRows


def canonical_json(payload: object) -> str:
    """Return the canonical JSON spelling durable identities use: sorted, compact, finite."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


_HEX64 = r"[0-9a-f]{64}"

type Sha256Hex = Annotated[str, Field(pattern=rf"^{_HEX64}$")]
type NonBlank = Annotated[str, Field(pattern=r"\S")]
type PathSafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")]
type Finite = Annotated[float, Field(allow_inf_nan=False)]
type PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type UnitInterval = Annotated[float, Field(ge=0, le=1)]
type PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
# Declared unboundedness: a window that never closes is stated in a word, never in a float.
UNBOUNDED: Literal["unbounded"] = "unbounded"
type Window = PositiveSeconds | Literal["unbounded"]
# Declared container resources, shaped like ``Window``: "no limit" is a word, never silence.
type Cores = PositiveFinite | Literal["unbounded"]
type WorkingSet = Annotated[int, Field(gt=0)] | Literal["unbounded"]
# A declared token allowance, the fourth member of the same family; the gateway meters a grant
# of this shape.
type Ceiling = int | Literal["unbounded"]
type NonEmpty[C] = Annotated[C, Field(min_length=1)]
# A set whose JSON spelling is its sorted members. A bare `frozenset` iterates strings in
# `PYTHONHASHSEED` order, so it spells one value differently in every process, and an event id
# is the content hash of those bytes. Annotate any set reaching a durable payload with this.
type SortedSet[T] = Annotated[
    frozenset[T],
    PlainSerializer(sorted, return_type=list, when_used="always"),
]


# A `str` subclass because Git refs, SQLite keys and JSON values consume an `Ident` directly.
class Ident(str):
    """Validate immutable string identities at construction."""

    __slots__ = ()
    # Coordinate names allow separators that ordinary path-safe names reject.
    NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
    COORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}")
    PATTERN = NAME

    def __new__(cls, value: str) -> Self:
        """Validate and construct one concrete identity."""
        if isinstance(value, Ident) and type(value) is not cls:
            msg = f"{cls.__name__} requires explicit string laundering from {type(value).__name__}"
            raise TypeError(msg)
        if cls is Ident or not isinstance(value, str) or not cls.PATTERN.fullmatch(value):
            msg = f"invalid {cls.__name__}: {value!r}"
            raise ValueError(msg)
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        """Return a representation naming the concrete identity class."""
        return f"{type(self).__name__}({str.__repr__(self)})"

    @classmethod
    # pydantic's hook for a core schema; without it `Ident` validates as a bare `str`.
    def __get_pydantic_core_schema__(
        cls,
        source: object,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Return the schema exposing this type's strict and coercing validation paths."""
        constructed = core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
        return core_schema.json_or_python_schema(
            json_schema=constructed,
            python_schema=core_schema.lax_or_strict_schema(
                lax_schema=constructed,
                strict_schema=core_schema.is_instance_schema(cls),
            ),
        )


class RunId(Ident):
    """One run; also the run's path-safe file stem."""


class NodeId(Ident):
    """Identify a search-tree node by its Git commit hash, from which Git parents give lineage."""

    PATTERN = re.compile(rf"[0-9a-f]{{40}}|{_HEX64}")


class TreeId(Ident):
    """Identify a Git TREE object, `git rev-parse <commit>:<path>`, never the commit itself."""

    PATTERN = NodeId.PATTERN


class EventId(Ident):
    """Identify a ledger event by the Secure Hash Algorithm 256 hash of its canonical payload."""


class ActionId(Ident):
    """Identify a unit of work reserved before dispatch."""


class ArtifactOid(Ident):
    """Address stored bytes by their lowercase Secure Hash Algorithm 256 hash."""

    PATTERN = re.compile(_HEX64)

    @classmethod
    def of_bytes(cls, data: bytes) -> ArtifactOid:
        """Return a payload's address, its lowercase Secure Hash Algorithm 256 digest."""
        return cls(hashlib.sha256(data).hexdigest())


class CriterionCoordinate(Ident):
    """Name one component of a run's evaluation standard, whose current value is a pin."""

    PATTERN = Ident.COORD


class BindingTag(Ident):
    """Name one opaque declared execution binding and accounting identity."""

    PATTERN = Ident.COORD


# Longest relative path this boundary accepts: Git trees, manifests, and mounts carry it whole.
_MAX_RELPATH_CHARS = 512


def require_safe_relpath(*, path: str) -> None:
    """Reject a path that could escape or alter a managed workspace."""
    if (
        not path
        or len(path) > _MAX_RELPATH_CHARS
        or "\x00" in path
        or "\\" in path
        or path.startswith("/")
    ):
        msg = f"unsafe relative path: {path!r}"
        raise ValueError(msg)
    if any(part in {"", ".", "..", ".git"} for part in path.split("/")):
        msg = f"unsafe relative path: {path!r}"
        raise ValueError(msg)


class InfraWeather(Exception):
    """Report a retryable provider, transport, or sandbox failure."""

    weathered_spend: SpendRows | None = None
    weathered_dialogue: bytes | None = None
    #: WHOSE FAULT THIS WAS, AND THEREFORE WHO PAYS FOR IT.
    #:
    #: `False` is the ordinary reading and the one this class was written for: nothing about the
    #: candidate was observed, so no candidate is measured by it and the run forfeits the slot.
    #:
    #: `True` says the candidate's own execution caused it -- an out-of-memory kill in its own
    #: container, a filled disk, a sandbox it broke. Those score `0.0` rather than forfeiting, and
    #: so does every ambiguous case: a candidate that could escape a bad score by killing its own
    #: sandbox would have found a strategy, and a search rewards what it measures.
    #:
    #: Only a demonstrable provider failure stays `False` under that rule -- a gateway that
    #: refused, timed out, or reset, which no candidate could have caused from inside a container
    #: that cannot reach the network except through it.
    by_candidate: bool = False
    #: WHETHER ASKING AGAIN CAN CHANGE THE ANSWER. `True` is weather: a refusal, a timeout, a
    #: reset, all of which the gateway retries.
    #: `False` is a verdict on the request itself -- a content filter refusing the prompt --
    #: which the same request would earn again, so it is answered to the episode as it stands.
    retryable: bool = True
    #: WHAT THE ROUTE ANSWERED, WHERE IT ANSWERED AT ALL. A refusal that reached the gateway as an
    #: exception carries no status. 429 says the IP is the limit and a reset socket does not, so the
    #: two are recorded apart.
    status: int = 0
    #: THE ROUTE'S OWN STATED WAIT, in seconds, where it stated one. Carried so the window the
    #: route asked for is the window it gets, rather than a backoff invented on this side.
    retry_after: float | None = None
    #: Safe classification and wire timing belong to this refusal, not a provider's last reply.
    refusal_kind: str | None = None
    dispatched_at: float | None = None
    elapsed: float | None = None
    #: WHOSE REFUSAL THIS IS. `False` is the route's own answer. `True` says THIS SIDE gave up --
    #: the retry budget declining to fund another attempt is the case that matters. Such a refusal
    #: is not the endpoint's and does not narrow its window.
    local: bool = False

    def __init__(self, *args: object) -> None:
        """Create a retryable infrastructure failure."""
        super().__init__(*args)

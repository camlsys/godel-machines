# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Detach mapping fields on frozen declarations without changing their bytes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import AfterValidator, GetCoreSchemaHandler, WrapSerializer
from pydantic_core import core_schema

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from pydantic_core.core_schema import CoreSchema, SerializerFunctionWrapHandler


def freeze_mapping[K, V](value: Mapping[K, V], /) -> Mapping[K, V]:
    """Return a read-only mapping copied off its caller, so a later write raises."""
    return MappingProxyType(dict(value))


def thaw_frozen[T](value: T, /) -> T | dict[Any, Any]:
    """Hand an already frozen mapping back as a plain dict, so re-validation stays in C.

    pydantic-core admits a non-dict mapping through the Python-level ``collections.abc.Mapping``
    instance check, and a signal handler that raises there is swallowed ("Exception ignored")
    and reported as ``dict_type``. Every ``dataclasses.replace`` re-validates the frozen fields,
    so without this a SIGINT during one surfaces as a ``ValidationError`` instead of an interrupt.
    """
    return dict(value) if type(value) is MappingProxyType else value


class _ThawFrozen:
    """Python-mode input only: JSON input is never a mapping proxy, and strict JSON decoding of
    nested values (a frozenset read from a JSON array) must stay in JSON mode."""

    def __get_pydantic_core_schema__(self, source: object, handler: GetCoreSchemaHandler) -> CoreSchema:
        inner = handler(source)
        return core_schema.json_or_python_schema(
            json_schema=inner,
            python_schema=core_schema.no_info_before_validator_function(thaw_frozen, inner),
            serialization=inner.get("serialization"),
        )


def serialize_frozen_mapping[K, V](
    value: Mapping[K, V],
    handler: SerializerFunctionWrapHandler,
) -> dict[K, V]:
    """Serialize through the declared schemas, so the bytes match the unfrozen spelling."""
    return handler(dict(value))


# Validates an ordinary mapping, then detaches it before the record is visible. A mapping it
# froze itself re-enters as the plain dict it was copied from.
type FrozenMapping[K, V] = Annotated[
    Mapping[K, V],
    _ThawFrozen(),
    AfterValidator(freeze_mapping),
    WrapSerializer(serialize_frozen_mapping),
]


def freeze_fields(record: DataclassInstance, /, *names: str) -> None:
    """Freeze named mapping fields in place, from ``__post_init__``, before any caller looks."""
    for name in names:
        # The documented way past a frozen dataclass's own refusing `__setattr__`.
        object.__setattr__(record, name, freeze_mapping(getattr(record, name)))


def frozen_default() -> Mapping[Any, Any]:
    """Return the empty read-only default, because pydantic never validates ``default_factory``."""
    return MappingProxyType({})

# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Canonical declared-resource rows and their one weighting formula."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import AfterValidator, ConfigDict, Field, TypeAdapter

from gm.ids import BindingTag, PathSafeName

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

type AxisName = PathSafeName
type Amount = Annotated[int | float, Field(ge=0, allow_inf_nan=False)]
type PositiveAmount = Annotated[int | float, Field(gt=0, allow_inf_nan=False)]
type Weight = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type SpendRow = tuple[BindingTag | None, AxisName, Amount]

_ROW = TypeAdapter(SpendRow, config=ConfigDict(strict=True))


def canonical(entries: Iterable[SpendRow]) -> tuple[SpendRow, ...]:
    """Return canonical rows: equal keys merged, zero totals removed, order imposed."""
    totals: dict[tuple[BindingTag | None, AxisName], int | float] = {}
    for entry in entries:
        tag, axis, amount = _ROW.validate_python(entry)
        totals[tag, axis] = totals.get((tag, axis), 0) + amount
    return tuple(
        (tag, axis, amount)
        for (tag, axis), amount in sorted(
            totals.items(),
            key=lambda item: (
                "" if item[0][0] is None else str(item[0][0]),
                item[0][1],
            ),
        )
        if amount != 0
    )


def _require_canonical(rows: tuple[SpendRow, ...]) -> tuple[SpendRow, ...]:
    if rows != canonical(rows):
        msg = "resource rows must be unique, nonzero, and in canonical order"
        raise ValueError(msg)
    return rows


type SpendRows = Annotated[tuple[SpendRow, ...], AfterValidator(_require_canonical)]


def merged(*many: SpendRows) -> tuple[SpendRow, ...]:
    """Return the canonical sum of any number of row collections."""
    return canonical(row for rows in many for row in rows)


def weighted(
    *,
    entries: Iterable[SpendRow],
    weights: Mapping[BindingTag, Mapping[AxisName, float]],
) -> dict[AxisName, float]:
    """Return per-axis totals with each caller-vouched per-tag weight applied exactly once."""
    totals: dict[AxisName, float] = {}
    for tag, axis, amount in entries:
        factor = 1.0
        factors = None if tag is None else weights.get(tag)
        if factors is not None:
            # The mapping declares the tag, so silence about one of its axes is
            # an omission, not a rate. A declared zero is a value; absence isn't.
            declared = factors.get(axis)
            if declared is None:
                msg = (
                    f"weights declare {tag} but set no rate for {tag}/{axis};"
                    " absence may not price a resource"
                )
                raise ValueError(msg)
            factor = declared
        totals[axis] = totals.get(axis, 0.0) + factor * amount
    return totals

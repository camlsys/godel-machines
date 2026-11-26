# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Define one validated `Ident` subclass per kind of name a domain manifest uses."""

from __future__ import annotations

from gm.ids import Ident


class DomainId(Ident):
    """Identify one composed problem setting and its run, one per store."""


class RoleId(Ident):
    """Identify a class of producers, each declaring an ordered task pool."""


class TaskId(Ident):
    """Identify a task in a role's pool, or the set name when the role declares none."""


class SampleKey(Ident):
    """Identify a sample by a host-only key, with slash and colon separators allowed."""

    PATTERN = Ident.COORD


class DatasetVersion(Ident):
    """Identify an immutable published build, changed by moving the criterion pin."""


class EvaluatorSlot(Ident):
    """Identify a judging position: a fixed host instrument, or a candidate consulting seat."""


class ScoringRuleId(Ident):
    """Identify the pinned rule behind a raw score, so a stored verdict stays readable."""

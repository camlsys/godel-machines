# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The imo form's scoring law, settled on the shared wire."""

from __future__ import annotations

import pathlib
import sys

FORM = pathlib.Path(__file__).parent
sys.path.insert(0, str(FORM.parents[2] / "common"))

import settlement

GRADE = "grade.txt"
POINTS = "points"
LOW, HIGH = 0, 7
UNREADABLE = (
    "the grading seat returned no readable grade, so there is no verdict about"
    " the writer to settle; a judgment that could not be made is not a zero"
)


def mark(*, score: object) -> int:
    """Return one mark as the exact integer it must be, refusing everything else."""
    if isinstance(score, bool) or not isinstance(score, int) or not LOW <= score <= HIGH:
        raise ValueError(f"a mark is an exact {LOW}-{HIGH} integer, not {score!r}")
    return score


def agreement(*, grade: object, human: float) -> float:
    """Return 1.0 when the grade is exactly the human mark, else 0.0; a malformed grade agrees with nothing."""
    return 0.0 if grade == settlement.MALFORMED else float(mark(score=grade) == human)


def raw_score(*, slot: str, request: dict) -> float:
    """Return this form's raw score for one judgment."""
    if slot != "marks":
        score = request["score"]
        if score == settlement.MALFORMED:
            raise ValueError(UNREADABLE)
        return mark(score=score)
    artifact = settlement.staged(request=request, key=GRADE)
    if artifact is None:
        raise ValueError(f"the marks anchor was staged {sorted(request['files'])}, not {GRADE!r}")
    graded = settlement.stated(artifact=artifact)
    return agreement(grade=graded, human=float(settlement.channel(request=request, key=POINTS)))


def meant(*, slot: str, raw: float, member: dict, form: dict) -> float:
    """Return what a raw score is worth, read off the slot's own declared interpretation."""
    kind = form["interpretations"][form["judges"][slot]["interpretation"]]
    return settlement.utility(kind=kind, raw=raw)


LAW = settlement.Law(form=FORM, raw_score=raw_score, meant=meant)
settled, main, WEATHER_EXIT = LAW.settled, LAW.main, settlement.WEATHER_EXIT

if __name__ == "__main__":
    main()

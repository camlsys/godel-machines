# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The toy form's scoring law, settled on the shared wire."""

from __future__ import annotations

import pathlib
import sys

FORM = pathlib.Path(__file__).parent
sys.path.insert(0, str(FORM.parents[2] / "common"))

import settlement

VERDICT = "verdict.txt"


def raw_score(*, slot: str, request: dict) -> float:
    """Return this form's raw score for one judgment."""
    if slot != "checker":
        return float(request["score"])
    artifact = settlement.staged(request=request, key=VERDICT)
    if artifact is None:
        raise ValueError(f"the checker was staged {sorted(request['files'])}, not {VERDICT!r}")
    channels = request["channels"]
    correct = float(channels["probe"] == channels["expected"])
    return 1.0 - abs(settlement.stated_score(artifact=artifact) - correct)


def meant(*, slot: str, raw: float, member: dict, form: dict) -> float:
    """Return what a raw score is worth: on this form the raw is already on `[0, 1]`."""
    return raw


LAW = settlement.Law(form=FORM, raw_score=raw_score, meant=meant)
settled, main, WEATHER_EXIT = LAW.settled, LAW.main, settlement.WEATHER_EXIT

if __name__ == "__main__":
    main()

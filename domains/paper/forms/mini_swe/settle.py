# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""The paper form's scoring law, settled on the shared wire."""

from __future__ import annotations

import pathlib
import sys

FORM = pathlib.Path(__file__).parent
sys.path.insert(0, str(FORM.parents[2] / "common"))

import settlement

VERDICT = "verdict.txt"
DECISION = "decision"
ACCEPT = "accept"


def resolved(*, request: dict, key: str) -> str:
    """Return one column the host resolved for this row; on this form an absent one is weather."""
    if not request["files"]:
        raise settlement.HostWeatherError("the host staged no bundle at all")
    try:
        return settlement.channel(request=request, key=key)
    except KeyError as unresolved:
        raise settlement.HostWeatherError(str(unresolved)) from unresolved


def raw_score(*, slot: str, request: dict) -> float:
    """Return this form's raw score for one judgment."""
    if slot == "apres":
        decision = resolved(request=request, key=DECISION)
        verdict = settlement.staged(request=request, key=VERDICT)
        if verdict is None:
            raise ValueError(
                f"the apres anchor was staged {sorted(request['files'])}, not {VERDICT!r}",
            )
        agreed = float(settlement.stated_score(artifact=verdict) == float(decision == ACCEPT))
        return agreed, {}, {DECISION: decision}
    return float(request["score"])


def meant(*, slot: str, raw: float, member: dict, form: dict) -> float:
    """Return what a raw score is worth on `[0, 1]`: an unreadable verdict is worth nothing."""
    return 0.0 if raw == settlement.MALFORMED else raw


LAW = settlement.Law(form=FORM, raw_score=raw_score, meant=meant)
settled, main, WEATHER_EXIT = LAW.settled, LAW.main, settlement.WEATHER_EXIT

if __name__ == "__main__":
    main()

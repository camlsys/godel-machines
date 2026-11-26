# Copyright 2026 The RQGM Authors
# SPDX-License-Identifier: Apache-2.0
"""What every form's settlement shares: the wire, the answer shape, the interpretation kinds."""

from __future__ import annotations

import base64
import binascii
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

MEMBER = "member"
MALFORMED = -1.0
WEATHER_EXIT = 75


class HostWeatherError(Exception):
    """The host failed to stage what this settlement grades; the candidate is not at fault."""


def utility(*, kind: dict[str, Any], raw: float) -> float:
    """Return what one declared interpretation makes of a raw score."""
    match kind:
        case {"kind": "threshold", "bar": bar}:
            return float(raw >= bar)
        case {"kind": "full_credit"}:
            return float(raw == 1.0)
        case {"kind": "band", "low": low, "high": high}:
            return float(low <= raw <= high)
        case {"kind": "graded"}:
            return min(max(raw, 0.0), 1.0)
    raise ValueError(f"undeclared interpretation kind: {kind}")


def decoded(*, staged: str, key: str) -> bytes:
    """Return one staged artifact's bytes off the wire's base64, or refuse as weather."""
    try:
        return base64.b64decode(staged, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise HostWeatherError(f"the staged {key!r} is not decodable off this wire: {exc}") from exc


def staged(*, request: dict, key: str) -> bytes | None:
    """Return one staged file's bytes, or None where the bundle carries no such file."""
    artifact = request["files"].get(key)
    return None if artifact is None else decoded(staged=artifact, key=key)


def channel(*, request: dict, key: str) -> str:
    """Return one resolved column of the corpus row, refusing by name where none was resolved."""
    resolved = request["channels"]
    if key not in resolved:
        raise KeyError(
            f"the host resolved no {key!r} column for this sample; it staged {sorted(resolved)}",
        )
    return resolved[key]


def stated(*, artifact: bytes) -> object:
    """Return the score a seat stated in its verdict artifact, `{"score": x}`, as it stated it."""
    verdict = json.loads(artifact.decode())
    if not isinstance(verdict, dict) or "score" not in verdict:
        raise ValueError(f"verdict artifact states no score: {verdict!r}")
    return verdict["score"]


def stated_score(*, artifact: bytes) -> float:
    """Return the stated score as a number."""
    score = stated(artifact=artifact)
    if not isinstance(score, int | float | str):
        raise ValueError(f"verdict score is not a number: {score!r}")
    return float(score)


class Law:
    """One form's settlement: its scoring law and meaning, run on the shared wire."""

    def __init__(
        self,
        *,
        form: Path,
        raw_score: Callable[..., Any],
        meant: Callable[..., float],
        weather: tuple[type[Exception], ...] = (HostWeatherError,),
    ) -> None:
        """Bind one form's law to the folder its `form.json` sits in."""
        self.form, self.raw_score, self.meant, self.weather = form, raw_score, meant, weather

    def settled(self, *, slot: str, form: dict, member: dict) -> dict:
        """Settle one member: the seven payload keys the kernel reads, plus the member."""
        named = member.get(MEMBER)
        if not isinstance(named, str) or not named.strip():
            raise ValueError(f"a member of this batch names no {MEMBER}: {named!r}")
        try:
            scored = self.raw_score(slot=slot, request=member)
            raw, evidence, labels = (
                (*scored, {}, {})[:3] if isinstance(scored, tuple) else (scored, {}, {})
            )
            outcome = self.meant(slot=slot, raw=raw, member=member, form=form)
        except (ArithmeticError, LookupError, TypeError, ValueError) as exc:
            raise ValueError(f"this settlement refused the member {named!r}: {exc}") from exc
        return {
            MEMBER: named,
            "score": raw,
            "outcome": outcome,
            "rule": form["judges"][slot]["rule"],
            "evidence": {
                name: base64.b64encode(value).decode("ascii") for name, value in evidence.items()
            },
            "utilities": {
                name: utility(kind=kind, raw=raw)
                for name, kind in form["interpretations"].items()
            },
            "units": {"evaluations": 1},
            "labels": labels,
        }

    def main(self) -> None:
        """Run the settlement: the slot from argv, the membership from stdin, answers to stdout."""
        slot = sys.argv[1]
        members = json.load(sys.stdin)
        if not isinstance(members, list):
            raise TypeError(
                f"this settlement was handed {type(members).__name__}, not a membership",
            )
        try:
            form = json.loads((self.form / "form.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"the {slot} settlement read no form.json beside it: {exc}")
            raise SystemExit(WEATHER_EXIT) from exc
        answers = []
        for member in members:
            try:
                answers.append(self.settled(slot=slot, form=form, member=member))
            except self.weather as exc:
                named = member.get(MEMBER) if isinstance(member, dict) else None
                sys.stderr.write(
                    f"the {slot} settlement measured no candidate for member {named!r}: {exc}",
                )
                raise SystemExit(WEATHER_EXIT) from exc
        sys.stdout.write(json.dumps(answers, sort_keys=True, separators=(",", ":")))

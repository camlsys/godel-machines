# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""A domain's corpus: one immutable dataset build per declared version, read from `data/fetch.json` and its CSV sets."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from gm.domain.identity import DatasetVersion, SampleKey, TaskId
from gm.domain.results import Artifact, InlineArtifact

type SampleSetName = str

#: The channel name that stages a column positionally, as the sample's input files.
INPUT = "input"
_MAX_FIELD = 2097152


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSpec:
    """One immutable dataset build: its named sets, and each sample's inputs, image, channels and source row."""

    version: DatasetVersion
    splits: Mapping[SampleSetName, tuple[SampleKey, ...]]
    inputs: Mapping[SampleKey, tuple[Artifact, ...]] = field(default_factory=dict)
    images: Mapping[SampleKey, str] = field(default_factory=dict)
    channels: Mapping[SampleKey, Mapping[str, bytes]] = field(default_factory=dict)
    source_rows: Mapping[SampleKey, Mapping[str, bytes]] = field(default_factory=dict)
    successors: tuple[DatasetSpec, ...] = ()

    def build(self, *, version: str) -> DatasetSpec:
        """Return the declared build one ``dataset`` pin names."""
        return next(one for one in (self, *self.successors) if str(one.version) == version)

    def rows(self) -> dict[SampleSetName, tuple[SampleKey, ...]]:
        """Return every set's membership across every build this corpus declares."""
        merged: dict[SampleSetName, tuple[SampleKey, ...]] = {}
        for build in (self, *self.successors):
            for name, keys in build.splits.items():
                merged[name] = (*merged.get(name, ()), *keys)
        return merged


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainSpec:
    """The sample set each task trains on when an expansion submits training."""

    task_sets: Mapping[TaskId, SampleSetName]

    @property
    def every(self) -> frozenset[SampleSetName]:
        """Every set any task trains on."""
        return frozenset(self.task_sets.values())


def _rows(path: Path) -> list[dict[str, str]]:
    restore = csv.field_size_limit(_MAX_FIELD)
    try:
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    finally:
        csv.field_size_limit(restore)


def _build(*, data: Path, declared: Mapping, version: str, sets: Mapping[str, str], successors=()) -> DatasetSpec:
    key, image = declared["key"], declared.get("image")
    columns = dict(declared["channels"])
    private = frozenset(declared.get("private", ()))
    splits: dict[str, tuple[SampleKey, ...]] = {}
    inputs: dict[SampleKey, tuple[Artifact, ...]] = {}
    channels: dict[SampleKey, dict[str, bytes]] = {}
    rows: dict[SampleKey, dict[str, bytes]] = {}
    for name, member in sorted(sets.items()):
        keys: list[SampleKey] = []
        for row in _rows(data / member):
            sample = SampleKey(row[key])
            keys.append(sample)
            encoded = {column: value.encode() for column, value in row.items() if value is not None}
            rows.setdefault(sample, {}).update({c: v for c, v in encoded.items() if c not in private})
            for column, channel in columns.items():
                if (value := encoded.get(column)) is None:
                    continue
                if channel == INPUT:
                    inputs[sample] = (InlineArtifact(data=value),)
                else:
                    channels.setdefault(sample, {})[channel] = value
        splits[name] = tuple(keys)
    return DatasetSpec(
        version=DatasetVersion(version),
        splits=splits,
        inputs=inputs,
        images={sample: row[image].decode() for sample, row in channels.items() if image in row},
        channels=channels,
        source_rows=rows,
        successors=tuple(successors),
    )


def read_dataset(data: Path) -> DatasetSpec:
    """Return the build a domain's `data/` folder declares, with its successor builds; refuse unbuilt data."""
    declared = json.loads((data / "fetch.json").read_text(encoding="utf-8"))
    sets = declared.get("sets") or {}
    if missing := sorted(name for name in sets.values() if not (data / name).is_file()):
        raise SystemExit(
            f"{data} lacks {', '.join(missing)}: build this domain's data first with"
            f" `python {declared['build']['tool']}` (see README.md)"
        )
    later = declared.get("successors") or {}
    successors = [_build(data=data, declared=declared, version=v, sets=later[v]) for v in sorted(later)]
    return _build(data=data, declared=declared, version=declared["version"], sets=sets, successors=successors)

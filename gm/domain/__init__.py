# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""A domain as the kernel reads it: the form manifest, its corpus, its standard, and the results it settles."""

from __future__ import annotations

from gm.domain.dataset import DatasetSpec, SampleSetName, TrainSpec
from gm.domain.identity import DatasetVersion, DomainId, EvaluatorSlot, RoleId, SampleKey, ScoringRuleId, TaskId
from gm.domain.manifest import (
    BindingConfig,
    CoordinateDeclarations,
    DomainManifest,
    EpisodeBinding,
    FixedInstrument,
    GeneratorSpec,
    ScorerSpec,
    StageSpec,
)
from gm.domain.results import (
    Artifact,
    ArtifactRef,
    EvaluatorJudgment,
    ExpansionOutcome,
    FailureReason,
    InlineArtifact,
    InvalidJudgment,
    JudgmentResult,
    TaskFailure,
    TaskResult,
    TaskSuccess,
)
from gm.domain.seams import ProcessInstrument, ScoredMember
from gm.domain.standard import (
    CompositeObjective,
    Curriculum,
    JudgedWork,
    MintedSet,
    PoolCriterion,
    PoolMint,
    SurfaceSide,
    TransitionFacts,
    TransitionPolicy,
    TransitionSpec,
    UtilitySurface,
    minted_origin,
)

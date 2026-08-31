from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .chunk import ChunkModel
from .hybrid import PointId, ScoredChunk


ChunkT = TypeVar("ChunkT", bound=ChunkModel)
CandidateKey = tuple[str, PointId]


@dataclass(frozen=True, slots=True)
class FusedChunk(Generic[ChunkT]):
    """A search candidate ranked by reciprocal rank fusion."""

    candidate: ScoredChunk[ChunkT]
    score: float
    dense_rank: int | None
    fulltext_rank: int | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("score must be finite and non-negative")
        for name, rank in (
            ("dense_rank", self.dense_rank),
            ("fulltext_rank", self.fulltext_rank),
        ):
            if rank is not None and rank <= 0:
                raise ValueError(f"{name} must be positive")


def _candidate_key(candidate: ScoredChunk[ChunkT]) -> CandidateKey:
    return candidate.collection_name, candidate.point_id


def _rank_by_score(
    candidates: Sequence[tuple[ScoredChunk[ChunkT], float]],
) -> dict[CandidateKey, int]:
    if any(not math.isfinite(score) for _, score in candidates):
        raise ValueError("ranked scores must be finite")
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item[1],
            item[0].collection_name,
            str(item[0].point_id),
        ),
    )
    return {
        _candidate_key(candidate): rank
        for rank, (candidate, _) in enumerate(ordered, start=1)
    }


def rrf_fuse(
    candidates: Sequence[ScoredChunk[ChunkT]],
    *,
    k: float,
    dense_weight: float,
    fulltext_weight: float,
    limit: int | None = None,
) -> list[FusedChunk[ChunkT]]:
    """Fuse Dense and refined full-text ranks for a union candidate list."""
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("k must be finite and non-negative")
    if not math.isfinite(dense_weight) or dense_weight < 0.0:
        raise ValueError("dense_weight must be finite and non-negative")
    if not math.isfinite(fulltext_weight) or fulltext_weight < 0.0:
        raise ValueError("fulltext_weight must be finite and non-negative")
    if dense_weight + fulltext_weight <= 0.0:
        raise ValueError("at least one weight must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    by_key: dict[CandidateKey, ScoredChunk[ChunkT]] = {}
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in by_key:
            raise ValueError(
                "candidates must be unique by collection_name and point_id"
            )
        if candidate.dense_score is None and candidate.fulltext_score is None:
            raise ValueError("each candidate must have at least one ranked score")
        by_key[key] = candidate

    dense_ranks = _rank_by_score(
        [
            (candidate, candidate.dense_score)
            for candidate in candidates
            if candidate.dense_score is not None
        ]
    )
    fulltext_ranks = _rank_by_score(
        [
            (candidate, candidate.fulltext_score)
            for candidate in candidates
            if candidate.fulltext_score is not None
        ]
    )
    fused: list[FusedChunk[ChunkT]] = []
    for key, candidate in by_key.items():
        dense_rank = dense_ranks.get(key)
        fulltext_rank = fulltext_ranks.get(key)
        score = 0.0
        if dense_rank is not None:
            score += dense_weight / (k + dense_rank)
        if fulltext_rank is not None:
            score += fulltext_weight / (k + fulltext_rank)
        fused.append(
            FusedChunk(
                candidate=candidate,
                score=score,
                dense_rank=dense_rank,
                fulltext_rank=fulltext_rank,
            )
        )

    fused.sort(
        key=lambda result: (
            -result.score,
            result.candidate.collection_name,
            str(result.candidate.point_id),
        )
    )
    return fused if limit is None else fused[:limit]

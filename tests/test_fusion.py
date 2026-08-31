from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from rakuhoku import ChunkModel, DenseField, ScoredChunk, SparseField, rrf_fuse


class TextChunk(ChunkModel):
    vectorization_schema_id: ClassVar[str] = "text-v1"

    text: str

    def dense_fields(self) -> Sequence[DenseField]:
        return [DenseField("text", self.text)]

    def sparse_fields(self) -> Sequence[SparseField]:
        return [SparseField(self.text)]

    def rerank_text(self) -> str:
        return self.text


def _candidate(
    point_id: str,
    *,
    dense_score: float | None,
    fulltext_score: float | None,
    collection_name: str = "documents",
) -> ScoredChunk[TextChunk]:
    return ScoredChunk(
        collection_name=collection_name,
        point_id=point_id,
        chunk=TextChunk(text=point_id),
        dense_score=dense_score,
        fulltext_score=fulltext_score,
    )


def test_rrf_fuse_builds_each_ranking_from_scored_chunks() -> None:
    candidates = [
        _candidate("a", dense_score=0.9, fulltext_score=0.8),
        _candidate("b", dense_score=0.8, fulltext_score=None),
        _candidate("c", dense_score=None, fulltext_score=0.9),
    ]

    fused = rrf_fuse(
        candidates,
        k=5.0,
        dense_weight=0.5,
        fulltext_weight=0.5,
    )

    assert [result.candidate.point_id for result in fused] == ["a", "c", "b"]
    assert fused[0].score == 0.5 / 6.0 + 0.5 / 7.0
    assert (fused[0].dense_rank, fused[0].fulltext_rank) == (1, 2)
    assert fused[1].score == 0.5 / 6.0
    assert (fused[1].dense_rank, fused[1].fulltext_rank) == (None, 1)


def test_rrf_fuse_uses_rrf_score_when_only_one_ranking_is_present() -> None:
    fused = rrf_fuse(
        [_candidate("a", dense_score=0.91, fulltext_score=None)],
        k=5.0,
        dense_weight=0.5,
        fulltext_weight=0.5,
    )

    assert fused[0].score == 0.5 / 6.0
    assert (fused[0].dense_rank, fused[0].fulltext_rank) == (1, None)


def test_rrf_fuse_uses_collection_name_as_part_of_candidate_identity() -> None:
    fused = rrf_fuse(
        [
            _candidate(
                "same", dense_score=0.9, fulltext_score=None, collection_name="a"
            ),
            _candidate(
                "same", dense_score=0.8, fulltext_score=None, collection_name="b"
            ),
        ],
        k=5.0,
        dense_weight=1.0,
        fulltext_weight=0.0,
    )

    assert [result.candidate.collection_name for result in fused] == ["a", "b"]


@pytest.mark.parametrize(
    ("k", "dense_weight", "fulltext_weight", "limit"),
    [
        (-1.0, 0.5, 0.5, None),
        (5.0, -0.5, 0.5, None),
        (5.0, 0.5, -0.5, None),
        (5.0, 0.0, 0.0, None),
        (5.0, 0.5, 0.5, -1),
    ],
)
def test_rrf_fuse_rejects_invalid_parameters(
    k: float,
    dense_weight: float,
    fulltext_weight: float,
    limit: int | None,
) -> None:
    with pytest.raises(ValueError):
        rrf_fuse(
            [_candidate("a", dense_score=0.9, fulltext_score=None)],
            k=k,
            dense_weight=dense_weight,
            fulltext_weight=fulltext_weight,
            limit=limit,
        )


def test_rrf_fuse_rejects_duplicate_or_unranked_candidates() -> None:
    candidate = _candidate("a", dense_score=0.9, fulltext_score=None)
    with pytest.raises(ValueError, match="unique"):
        rrf_fuse(
            [candidate, candidate],
            k=5.0,
            dense_weight=0.5,
            fulltext_weight=0.5,
        )

    with pytest.raises(ValueError, match="at least one ranked score"):
        rrf_fuse(
            [_candidate("a", dense_score=None, fulltext_score=None)],
            k=5.0,
            dense_weight=0.5,
            fulltext_weight=0.5,
        )

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from rag_vectorizers import ChunkModel, DenseField, SparseField


class ExampleChunk(ChunkModel):
    vectorization_schema_id: ClassVar[str] = "example-v1"

    title: str
    body: str
    fallback: str | None = None

    def dense_fields(self) -> Sequence[DenseField]:
        return (
            DenseField("title", self.title),
            DenseField("body", self.body or self.fallback or ""),
        )

    def sparse_fields(self) -> Sequence[SparseField]:
        return (
            SparseField(self.title, weight=2.0),
            SparseField(self.body or self.fallback or ""),
        )

    def rerank_text(self) -> str:
        return self.body or self.fallback or ""


def test_chunk_model_keeps_client_fields_and_vectorization_separate() -> None:
    chunk = ExampleChunk(title="規程", body="利用できない")

    assert chunk.model_dump(mode="json") == {
        "title": "規程",
        "body": "利用できない",
        "fallback": None,
    }
    assert chunk.vectorization_schema_id == "example-v1"
    assert chunk.dense_fields() == (
        DenseField("title", "規程"),
        DenseField("body", "利用できない"),
    )
    assert chunk.sparse_fields() == (
        SparseField("規程", weight=2.0),
        SparseField("利用できない"),
    )
    assert chunk.rerank_text() == "利用できない"


def test_chunk_model_methods_can_apply_client_fallbacks() -> None:
    chunk = ExampleChunk(title="規程", body="", fallback="自動生成された概要")

    assert chunk.dense_fields()[1] == DenseField("body", "自動生成された概要")
    assert chunk.sparse_fields()[1] == SparseField("自動生成された概要")
    assert chunk.rerank_text() == "自動生成された概要"


def test_chunk_model_remains_abstract_without_required_methods() -> None:
    class IncompleteChunk(ChunkModel):
        vectorization_schema_id: ClassVar[str] = "incomplete-v1"
        text: str

    with pytest.raises(TypeError, match="abstract"):
        IncompleteChunk(text="本文")  # pyright: ignore[reportAbstractUsage]


def test_dense_field_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        DenseField("  ", "本文")

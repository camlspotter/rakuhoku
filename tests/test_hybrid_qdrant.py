from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient

from rakuhoku import (
    ChunkModel,
    ChunkPoint,
    DenseField,
    HybridDBConfig,
    HybridQdrantDB,
    PreparedQuery,
    QueryVectorizer,
    SparseField,
    SparseVector,
)


class TextChunk(ChunkModel):
    vectorization_schema_id: ClassVar[str] = "text-v1"

    title: str
    text: str

    def dense_fields(self) -> Sequence[DenseField]:
        return (DenseField("title", self.title), DenseField("text", self.text))

    def sparse_fields(self) -> Sequence[SparseField]:
        return (SparseField(self.title, weight=2.0), SparseField(self.text))

    def rerank_text(self) -> str:
        return self.text


class OtherTextChunk(TextChunk):
    vectorization_schema_id: ClassVar[str] = "other-schema"


class FakeDenseEncoder:
    model_name = "fake-dense-v1"
    dimension = 2

    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        self.query_calls += 1
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        self.document_calls += 1
        return np.asarray(
            [
                [1.0, 0.0] if "dense-only" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class FakeSparseEncoder:
    algorithm_id = "fake-sparse-v1"

    def encode(self, value: str | Sequence[SparseField]) -> SparseVector:
        if isinstance(value, str):
            texts = [value]
        else:
            texts = [field.text for field in value]
        if any("sparse-only" in text or "利用できない" == text for text in texts):
            return SparseVector(indices=[7], values=[1.0])
        return SparseVector(indices=[8], values=[1.0])


def _components() -> tuple[FakeDenseEncoder, QueryVectorizer, HybridDBConfig]:
    dense = FakeDenseEncoder()
    vectorizer = QueryVectorizer(
        dense_encoder=dense,
        sparse_encoder=FakeSparseEncoder(),
    )
    config = HybridDBConfig(
        collection_name="chunks",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        dense_dimension=2,
        dense_model_id="fake-dense-v1",
        sparse_algorithm_id="fake-sparse-v1",
        vectorization_schema_id="text-v1",
    )
    return dense, vectorizer, config


def test_query_vectorizer_prepares_reusable_vectors() -> None:
    dense, vectorizer, _ = _components()

    prepared = vectorizer.encode("利用できない")

    assert prepared.text == "利用できない"
    assert prepared.dense_vector.tolist() == [1.0, 0.0]
    assert prepared.sparse_vector == SparseVector(indices=[7], values=[1.0])
    assert prepared.dense_model_id == "fake-dense-v1"
    assert prepared.sparse_algorithm_id == "fake-sparse-v1"
    assert dense.query_calls == 1


def test_one_db_search_unions_dense_and_sparse_and_reranks() -> None:
    async def run() -> None:
        client = AsyncQdrantClient(":memory:")
        dense, vectorizer, config = _components()
        try:
            db = await HybridQdrantDB.create(
                client=client,
                config=config,
                chunk_type=TextChunk,
                vectorizer=vectorizer,
            )
            await db.upsert_chunks(
                [
                    ChunkPoint(
                        id=1,
                        chunk=TextChunk(
                            title="dense-only", text="設備を利用できます"
                        ),
                    ),
                    ChunkPoint(
                        id=2,
                        chunk=TextChunk(
                            title="sparse-only", text="この設備は利用できない"
                        ),
                    ),
                ]
            )
            stored = await client.retrieve(
                collection_name="chunks", ids=[2], with_payload=True
            )
            assert stored[0].payload is not None
            assert stored[0].payload["_rakuhoku"] == {
                "dense_text": "title: sparse-only\ntext: この設備は利用できない",
                "sparse_fields": [
                    {"text": "sparse-only", "weight": 2.0},
                    {"text": "この設備は利用できない", "weight": 1.0},
                ],
                "rerank_text": "この設備は利用できない",
                "vectorization_schema_id": "text-v1",
            }

            results = await db.search(
                "利用できない", dense_limit=1, sparse_limit=1
            )

            by_id = {result.point_id: result for result in results}
            assert set(by_id) == {1, 2}
            assert by_id[1].dense_score == pytest.approx(1.0)
            assert by_id[2].dense_score is None
            assert by_id[2].fulltext_score > by_id[1].fulltext_score
            assert by_id[2].chunk == TextChunk(
                title="sparse-only", text="この設備は利用できない"
            )
            assert all(result.collection_name == "chunks" for result in results)
            assert dense.query_calls == 1
            assert dense.document_calls == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_search_by_vectors_reuses_a_prepared_query() -> None:
    async def run() -> None:
        client = AsyncQdrantClient(":memory:")
        dense, vectorizer, config = _components()
        try:
            db = await HybridQdrantDB.create(
                client=client,
                config=config,
                chunk_type=TextChunk,
                vectorizer=vectorizer,
            )
            await db.upsert_chunks(
                [
                    ChunkPoint(
                        id=10,
                        chunk=TextChunk(
                            title="dense-only", text="利用できない"
                        ),
                    )
                ]
            )
            prepared = vectorizer.encode("利用できない")

            first = await db.search_by_vectors(prepared, sparse_limit=0)
            second = await db.search_by_vectors(prepared, sparse_limit=0)

            assert [result.point_id for result in first] == [10]
            assert [result.point_id for result in second] == [10]
            assert dense.query_calls == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_open_validates_collection_metadata_and_vector_contract() -> None:
    async def run() -> None:
        client = AsyncQdrantClient(":memory:")
        _, vectorizer, config = _components()
        try:
            await HybridQdrantDB.create(
                client=client,
                config=config,
                chunk_type=TextChunk,
                vectorizer=vectorizer,
            )
            opened = await HybridQdrantDB.open(
                client=client,
                config=config,
                chunk_type=TextChunk,
                vectorizer=vectorizer,
            )
            assert opened.config == config

            incompatible = HybridDBConfig(
                collection_name="chunks",
                dense_vector_name="dense",
                sparse_vector_name="sparse",
                dense_dimension=2,
                dense_model_id="fake-dense-v1",
                sparse_algorithm_id="fake-sparse-v1",
                vectorization_schema_id="other-schema",
            )
            with pytest.raises(ValueError, match="metadata does not match"):
                await HybridQdrantDB.open(
                    client=client,
                    config=incompatible,
                    chunk_type=OtherTextChunk,
                    vectorizer=vectorizer,
                )
        finally:
            await client.close()

    asyncio.run(run())


def test_prepared_query_rejects_a_matrix() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        PreparedQuery(
            text="query",
            dense_vector=np.asarray([[1.0, 0.0]], dtype=np.float32),
            sparse_vector=SparseVector(indices=[], values=[]),
            dense_model_id="fake-dense-v1",
            sparse_algorithm_id="fake-sparse-v1",
        )

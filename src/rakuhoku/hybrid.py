from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from qdrant_client import AsyncQdrantClient, models

from .chunk import ChunkModel
from .rerank import RerankExplanation, SudachiLexicalReranker
from .types import SparseField


FloatVector = NDArray[np.float32]
PointId = int | str | UUID
ChunkT = TypeVar("ChunkT", bound=ChunkModel)
ChunkU = TypeVar("ChunkU", bound=ChunkModel)

COLLECTION_METADATA_KEY = "_rakuhoku" # XXX not required
SCHEMA_VERSION = 1
PAYLOAD_ADAPTER = TypeAdapter(dict[str, JsonValue])


class DenseEncoder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]: ...

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]: ...


class SparseEncoder(Protocol):
    @property
    def algorithm_id(self) -> str: ...

    def encode(
        self, value: str | Sequence[SparseField]
    ) -> models.SparseVector: ...


@dataclass(frozen=True, slots=True)
class PreparedDenseQuery:
    vector: FloatVector

    def __post_init__(self) -> None:
        dense = np.asarray(self.vector, dtype=np.float32)
        if dense.ndim != 1:
            raise ValueError("dense query vector must be one-dimensional")
        object.__setattr__(self, "vector", dense)


@dataclass(frozen=True, slots=True)
class PreparedSparseQuery:
    text: str
    vector: models.SparseVector


@dataclass(frozen=True, slots=True)
class PreparedSearch:
    """Multiple prepared dense and sparse queries for one search."""

    dense_queries: list[PreparedDenseQuery]
    sparse_queries: list[PreparedSparseQuery]
    dense_model_id: str
    sparse_algorithm_id: str


class QueryVectorizer:
    """Compute reusable dense and sparse query vectors."""

    def __init__(
        self, *, dense_encoder: DenseEncoder, sparse_encoder: SparseEncoder
    ) -> None:
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder

    @property
    def dense_model_id(self) -> str:
        return self.dense_encoder.model_name

    @property
    def sparse_algorithm_id(self) -> str:
        return self.sparse_encoder.algorithm_id

    @property
    def dense_dimension(self) -> int:
        return self.dense_encoder.dimension

    def prepare(
        self,
        *,
        dense_queries: Sequence[str],
        sparse_queries: Sequence[str],
    ) -> PreparedSearch:
        dense_texts = list(dense_queries)
        dense_vectors = (
            self.dense_encoder.encode_queries(dense_texts)
            if dense_texts
            else np.empty((0, self.dense_dimension), dtype=np.float32)
        )
        if dense_vectors.shape != (len(dense_texts), self.dense_dimension):
            raise RuntimeError("dense encoder returned an unexpected query shape")
        return PreparedSearch(
            dense_queries=[
                PreparedDenseQuery(vector=dense_vectors[index])
                for index in range(len(dense_texts))
            ],
            sparse_queries=[
                PreparedSparseQuery(
                    text=text,
                    vector=self.sparse_encoder.encode(text),
                )
                for text in sparse_queries
            ],
            dense_model_id=self.dense_model_id,
            sparse_algorithm_id=self.sparse_algorithm_id,
        )


@dataclass(frozen=True, slots=True)
class HybridDBConfig:
    collection_name: str
    dense_vector_name: str
    sparse_vector_name: str
    dense_dimension: int
    dense_model_id: str
    sparse_algorithm_id: str
    vectorization_schema_id: str

    def __post_init__(self) -> None:
        string_values = (
            self.collection_name,
            self.dense_vector_name,
            self.sparse_vector_name,
            self.dense_model_id,
            self.sparse_algorithm_id,
            self.vectorization_schema_id,
        )
        if any(not value.strip() for value in string_values):
            raise ValueError("DB configuration strings must not be empty")
        if self.dense_vector_name == self.sparse_vector_name:
            raise ValueError("dense and sparse vector names must differ")
        if self.dense_dimension <= 0:
            raise ValueError("dense_dimension must be positive")


@dataclass(frozen=True, slots=True)
class ChunkPoint(Generic[ChunkT]):
    id: PointId
    chunk: ChunkT


@dataclass(frozen=True, slots=True)
class ScoredChunk(Generic[ChunkT]):
    """A union candidate with dense, raw Sparse, and fused full-text scores."""

    collection_name: str
    point_id: PointId
    chunk: ChunkT
    dense_score: float | None
    fulltext_score: float | None
    sparse_score: float | None = None
    fulltext_explanation: RerankExplanation | None = None

    def __post_init__(self) -> None:
        if self.fulltext_score is not None:
            assert math.isfinite(self.fulltext_score), (
                f"fulltext_score must be finite, got {self.fulltext_score!r}"
            )
            assert 0.0 <= self.fulltext_score <= 1.0, (
                f"fulltext_score must be in [0, 1], got {self.fulltext_score!r}"
            )
        if self.sparse_score is not None:
            assert math.isfinite(self.sparse_score), (
                f"sparse_score must be finite, got {self.sparse_score!r}"
            )
            assert self.sparse_score >= 0.0, (
                f"sparse_score must be non-negative, got {self.sparse_score!r}"
            )


@dataclass(frozen=True, slots=True)
class _Candidate(Generic[ChunkT]):
    point_id: PointId
    chunk: ChunkT
    rerank_text: str
    dense_score: float | None


class _CollectionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    dense_model_id: str
    sparse_algorithm_id: str
    vectorization_schema_id: str
    dense_vector_name: str
    sparse_vector_name: str


class HybridQdrantDB(Generic[ChunkT]):
    """Standard async API for one dense+sparse Qdrant collection."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        config: HybridDBConfig,
        chunk_type: type[ChunkT],
        vectorizer: QueryVectorizer,
        reranker: SudachiLexicalReranker,
    ) -> None:
        self.client = client
        self.config = config
        self.chunk_type = chunk_type
        self.vectorizer = vectorizer
        self.reranker = reranker
        self._validate_local_contract()

    @classmethod
    async def create(
        cls,
        *,
        client: AsyncQdrantClient,
        config: HybridDBConfig,
        chunk_type: type[ChunkU],
        vectorizer: QueryVectorizer,
        reranker: SudachiLexicalReranker | None = None,
    ) -> HybridQdrantDB[ChunkU]:
        db = HybridQdrantDB[ChunkU](
            client=client,
            config=config,
            chunk_type=chunk_type,
            vectorizer=vectorizer,
            reranker=reranker or SudachiLexicalReranker(),
        )
        if await client.collection_exists(config.collection_name):
            raise ValueError(
                f"Qdrant collection {config.collection_name!r} already exists"
            )
        await client.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                config.dense_vector_name: models.VectorParams(
                    size=config.dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
            metadata=db._collection_metadata(),
        )
        return db

    @classmethod
    async def open(
        cls,
        *,
        client: AsyncQdrantClient,
        config: HybridDBConfig,
        chunk_type: type[ChunkU],
        vectorizer: QueryVectorizer,
        reranker: SudachiLexicalReranker | None = None,
    ) -> HybridQdrantDB[ChunkU]:
        db = HybridQdrantDB[ChunkU](
            client=client,
            config=config,
            chunk_type=chunk_type,
            vectorizer=vectorizer,
            reranker=reranker or SudachiLexicalReranker(),
        )
        if not await client.collection_exists(config.collection_name):
            raise ValueError(
                f"Qdrant collection {config.collection_name!r} does not exist"
            )
        info = await client.get_collection(config.collection_name)
        db._validate_collection(info)
        return db

    def _validate_local_contract(self) -> None:
        config = self.config
        if self.vectorizer.dense_model_id != config.dense_model_id:
            raise ValueError("dense encoder model does not match DB configuration")
        if self.vectorizer.sparse_algorithm_id != config.sparse_algorithm_id:
            raise ValueError("sparse encoder algorithm does not match DB configuration")
        if self.vectorizer.dense_dimension != config.dense_dimension:
            raise ValueError("dense encoder dimension does not match DB configuration")
        if self.chunk_type.vectorization_schema_id != config.vectorization_schema_id:
            raise ValueError("chunk schema does not match DB configuration")

    def _rakuhoku_metadata(self) -> _CollectionMetadata:
        config = self.config
        return _CollectionMetadata(
            schema_version=SCHEMA_VERSION,
            dense_model_id=config.dense_model_id,
            sparse_algorithm_id=config.sparse_algorithm_id,
            vectorization_schema_id=config.vectorization_schema_id,
            dense_vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
        )

    def _collection_metadata(self) -> dict[str, object]:
        return {
            COLLECTION_METADATA_KEY: self._rakuhoku_metadata().model_dump()
        }

    def _validate_collection(self, info: models.CollectionInfo) -> None:
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors
        if not isinstance(vectors, dict):
            raise ValueError("collection does not use named dense vectors")
        dense = vectors.get(self.config.dense_vector_name)
        if dense is None:
            raise ValueError("configured dense vector is missing from collection")
        if dense.size != self.config.dense_dimension:
            raise ValueError("collection dense vector dimension does not match")
        if dense.distance != models.Distance.COSINE:
            raise ValueError("collection dense vector must use cosine distance")
        if not isinstance(sparse_vectors, dict):
            raise ValueError("collection has no named sparse vectors")
        sparse = sparse_vectors.get(self.config.sparse_vector_name)
        if sparse is None:
            raise ValueError("configured sparse vector is missing from collection")
        if sparse.modifier != models.Modifier.IDF:
            raise ValueError("collection sparse vector must use Modifier.IDF")
        metadata: object = info.config.metadata
        if not isinstance(metadata, dict):
            raise ValueError("collection has no rakuhoku metadata")
        try:
            stored_metadata = _CollectionMetadata.model_validate(
                metadata.get(COLLECTION_METADATA_KEY)
            )
        except ValidationError as exc:
            raise ValueError("collection rakuhoku metadata is invalid") from exc
        if stored_metadata != self._rakuhoku_metadata():
            raise ValueError("collection rakuhoku metadata does not match")

    @staticmethod
    def _chunk_payload(chunk: ChunkModel) -> dict[str, JsonValue]:
        return PAYLOAD_ADAPTER.validate_python(chunk.model_dump(mode="json"))

    async def upsert_chunks(
        self, points: Sequence[ChunkPoint[ChunkT]], *, wait: bool = True
    ) -> None:
        if not points:
            return
        chunks = [point.chunk for point in points]
        for chunk in chunks:
            if not isinstance(chunk, self.chunk_type):
                raise TypeError(
                    f"expected {self.chunk_type.__name__}, got {type(chunk).__name__}"
                )
            if chunk.vectorization_schema_id != self.config.vectorization_schema_id:
                raise ValueError("chunk vectorization schema does not match collection")

        # Dense
        dense_texts = [chunk.dense_text() for chunk in chunks]
        dense_vectors = await asyncio.to_thread(
            self.vectorizer.dense_encoder.encode_documents, dense_texts
        )
        if dense_vectors.shape != (len(chunks), self.config.dense_dimension):
            raise RuntimeError("dense encoder returned an unexpected vector shape")

        # Sparse
        sparse_fields = [tuple(chunk.sparse_fields()) for chunk in chunks]
        sparse_vectors = await asyncio.to_thread(
            lambda: [
                self.vectorizer.sparse_encoder.encode(fields)
                for fields in sparse_fields
            ]
        )

        qdrant_points = [
            models.PointStruct(
                id=point.id,
                vector={
                    self.config.dense_vector_name: dense_vectors[index].tolist(),
                    self.config.sparse_vector_name: sparse_vectors[index],
                },
                payload=self._chunk_payload(point.chunk),
            )
            for index, point in enumerate(points)
        ]
        await self.client.upsert(
            collection_name=self.config.collection_name,
            points=qdrant_points,
            wait=wait,
        )

    async def delete_chunks(
        self, point_ids: Sequence[PointId], *, wait: bool = True
    ) -> None:
        if not point_ids:
            return
        await self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=wait,
        )

    async def get_chunks_by_filter(
        self,
        *,
        query_filter: models.Filter,
        limit: int = 50,
    ) -> list[ChunkPoint[ChunkT]]:
        """Return validated chunks selected only by a Qdrant payload filter."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []

        records, _ = await self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        chunks: list[ChunkPoint[ChunkT]] = []
        for record in records:
            if record.payload is None:
                raise ValueError(f"Qdrant point {record.id!r} has no payload")
            chunks.append(
                ChunkPoint(
                    id=record.id,
                    chunk=self.chunk_type.model_validate(record.payload),
                )
            )
        return chunks

    def _validate_prepared_search(self, search: PreparedSearch) -> None:
        if search.dense_model_id != self.config.dense_model_id:
            raise ValueError("query dense model does not match collection")
        if search.sparse_algorithm_id != self.config.sparse_algorithm_id:
            raise ValueError("query sparse algorithm does not match collection")
        if any(
            query.vector.shape != (self.config.dense_dimension,)
            for query in search.dense_queries
        ):
            raise ValueError("dense query vector dimension does not match collection")

    def _candidate_from_point(
        self,
        point: models.ScoredPoint,
        *,
        dense_score: float | None,
    ) -> _Candidate[ChunkT]:
        if point.payload is None:
            raise ValueError(f"Qdrant point {point.id!r} has no payload")
        chunk = self.chunk_type.model_validate(point.payload)
        point_id = point.id
        return _Candidate(
            point_id=point_id,
            chunk=chunk,
            rerank_text=chunk.rerank_text(),
            dense_score=dense_score,
        )

    async def search_by_vectors(
        self,
        search: PreparedSearch,
        *,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        query_filter: models.Filter | None = None,
        with_fulltext_explanation: bool = False,
    ) -> list[ScoredChunk[ChunkT]]:
        """Search one DB, union candidates, and add a lexical full-text score."""
        if dense_limit < 0 or sparse_limit < 0:
            raise ValueError("search limits must be non-negative")
        self._validate_prepared_search(search)
        if (
            (dense_limit == 0 or not search.dense_queries)
            and (sparse_limit == 0 or not search.sparse_queries)
        ):
            return []

        async def dense_search(
            query: PreparedDenseQuery,
        ) -> list[models.ScoredPoint]:
            response = await self.client.query_points(
                collection_name=self.config.collection_name,
                query=query.vector.tolist(),
                using=self.config.dense_vector_name,
                query_filter=query_filter,
                limit=dense_limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(response.points)

        async def sparse_search(
            query: PreparedSparseQuery,
        ) -> list[models.ScoredPoint]:
            if not query.vector.indices:
                return []
            response = await self.client.query_points(
                collection_name=self.config.collection_name,
                query=query.vector,
                using=self.config.sparse_vector_name,
                query_filter=query_filter,
                limit=sparse_limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(response.points)

        dense_calls = (
            [dense_search(query) for query in search.dense_queries]
            if dense_limit > 0
            else []
        )
        sparse_calls = (
            [sparse_search(query) for query in search.sparse_queries]
            if sparse_limit > 0
            else []
        )
        result_sets = await asyncio.gather(*(dense_calls + sparse_calls))
        dense_result_sets = result_sets[: len(dense_calls)]
        sparse_result_sets = (
            result_sets[len(dense_calls) :]
            if sparse_limit > 0
            else [[] for _ in search.sparse_queries]
        )

        candidates: dict[PointId, _Candidate[ChunkT]] = {}
        for result_set in dense_result_sets:
            for point in result_set:
                candidate = self._candidate_from_point(
                    point,
                    dense_score=float(point.score),
                )
                previous = candidates.get(candidate.point_id)
                if (
                    previous is None
                    or previous.dense_score is None
                    or (
                        candidate.dense_score is not None
                        and candidate.dense_score > previous.dense_score
                    )
                ):
                    candidates[candidate.point_id] = candidate
        for result_set in sparse_result_sets:
            for point in result_set:
                candidate = self._candidate_from_point(
                    point,
                    dense_score=None,
                )
                if candidate.point_id not in candidates:
                    candidates[candidate.point_id] = candidate

        candidate_list = list(candidates.values())
        rerank_texts = [candidate.rerank_text for candidate in candidate_list]
        rerank_task = asyncio.create_task(
            asyncio.to_thread(
                lambda: [
                    self.reranker.rerank(query.text, rerank_texts)
                    for query in search.sparse_queries
                ]
            )
        )

        async def complete_sparse_scores(
            query: PreparedSparseQuery,
            initial_results: Sequence[models.ScoredPoint],
        ) -> dict[PointId, float]:
            raw_scores = {
                point.id: float(point.score)
                for point in initial_results
            }
            unknown_ids = [
                candidate.point_id
                for candidate in candidate_list
                if candidate.point_id not in raw_scores
            ]
            if unknown_ids and query.vector.indices:
                point_filter = models.Filter(
                    must=[models.HasIdCondition(has_id=unknown_ids)]
                )
                restricted_filter = (
                    point_filter
                    if query_filter is None
                    else models.Filter(must=[query_filter, point_filter])
                )
                response = await self.client.query_points(
                    collection_name=self.config.collection_name,
                    query=query.vector,
                    using=self.config.sparse_vector_name,
                    query_filter=restricted_filter,
                    limit=len(unknown_ids),
                    with_payload=False,
                    with_vectors=False,
                )
                raw_scores.update(
                    (point.id, float(point.score))
                    for point in response.points
                )
            for point_id in unknown_ids:
                raw_scores.setdefault(point_id, 0.0)
            return raw_scores

        raw_score_sets = await asyncio.gather(
            *[
                complete_sparse_scores(query, result_set)
                for query, result_set in zip(
                    search.sparse_queries,
                    sparse_result_sets,
                    strict=True,
                )
            ]
        )
        reranked_sets = await rerank_task
        fulltext_scores: list[float | None] = [None for _ in candidate_list]
        sparse_scores: list[float | None] = [None for _ in candidate_list]
        explanations: list[RerankExplanation | None] = [
            None for _ in candidate_list
        ]
        config = self.reranker.config
        score_weight_total = (
            config.sparse_weight
            + config.order_weight
            + config.proximity_weight
        )
        for raw_scores, reranked in zip(
            raw_score_sets, reranked_sets, strict=True
        ):
            maximum_sparse_score = max(raw_scores.values(), default=0.0)
            for item in reranked:
                point_id = candidate_list[item.original_index].point_id
                sparse_score = raw_scores[point_id]
                normalized_sparse_score = (
                    sparse_score / maximum_sparse_score
                    if maximum_sparse_score > 0.0
                    else 0.0
                )
                fulltext_score = (
                    config.sparse_weight * normalized_sparse_score
                    + config.order_weight * item.explanation.order_score
                    + config.proximity_weight
                    * item.explanation.proximity_score
                ) / score_weight_total
                previous_score = fulltext_scores[item.original_index]
                if previous_score is None or fulltext_score > previous_score:
                    fulltext_scores[item.original_index] = fulltext_score
                    sparse_scores[item.original_index] = sparse_score
                    explanations[item.original_index] = item.explanation

        results = [
            ScoredChunk(
                collection_name=self.config.collection_name,
                point_id=candidate.point_id,
                chunk=candidate.chunk,
                dense_score=candidate.dense_score,
                fulltext_score=fulltext_scores[index],
                sparse_score=sparse_scores[index],
                fulltext_explanation=(
                    explanations[index] if with_fulltext_explanation else None
                ),
            )
            for index, candidate in enumerate(candidate_list)
        ]
        def result_sort_key(
            result: ScoredChunk[ChunkT],
        ) -> tuple[int, float, str]:
            if result.fulltext_score is not None:
                return (0, -result.fulltext_score, str(result.point_id))
            dense_score = result.dense_score
            assert dense_score is not None
            return (1, -dense_score, str(result.point_id))

        return sorted(results, key=result_sort_key)

    async def search(
        self,
        *,
        dense_queries: Sequence[str],
        sparse_queries: Sequence[str],
        dense_limit: int = 50,
        sparse_limit: int = 50,
        query_filter: models.Filter | None = None,
        with_fulltext_explanation: bool = False,
    ) -> list[ScoredChunk[ChunkT]]:
        """Standard one-DB search: encode, retrieve, union, and rerank."""
        prepared = await asyncio.to_thread(
            self.vectorizer.prepare,
            dense_queries=dense_queries,
            sparse_queries=sparse_queries,
        )
        return await self.search_by_vectors(
            prepared,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            query_filter=query_filter,
            with_fulltext_explanation=with_fulltext_explanation,
        )

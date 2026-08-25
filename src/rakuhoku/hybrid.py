from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from .chunk import ChunkModel
from .rerank import RerankExplanation, SudachiLexicalReranker
from .types import DenseField, SparseField, SparseVector


FloatVector = NDArray[np.float32]
PointId = int | str | UUID
ChunkT = TypeVar("ChunkT", bound=ChunkModel)
ChunkU = TypeVar("ChunkU", bound=ChunkModel)

COLLECTION_METADATA_KEY = "_rakuhoku"
PAYLOAD_METADATA_KEY = "_rakuhoku"
SCHEMA_VERSION = 1


class DenseEncoder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]: ...

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]: ...


class SparseEncoder(Protocol):
    @property
    def algorithm_id(self) -> str: ...

    def encode(self, value: str | Sequence[SparseField]) -> SparseVector: ...


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Dense and sparse representations computed once for reuse across DBs."""

    text: str
    dense_vector: FloatVector
    sparse_vector: SparseVector
    dense_model_id: str
    sparse_algorithm_id: str

    def __post_init__(self) -> None:
        dense = np.asarray(self.dense_vector, dtype=np.float32)
        if dense.ndim != 1:
            raise ValueError("dense_vector must be one-dimensional")
        object.__setattr__(self, "dense_vector", dense)


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

    def encode_dense(self, query: str) -> FloatVector:
        vectors = self.dense_encoder.encode_queries([query])
        return np.asarray(vectors[0], dtype=np.float32)

    def encode_sparse(self, query: str) -> SparseVector:
        return self.sparse_encoder.encode(query)

    def encode(self, query: str) -> PreparedQuery:
        return PreparedQuery(
            text=query,
            dense_vector=self.encode_dense(query),
            sparse_vector=self.encode_sparse(query),
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
    """A union candidate with independent dense and full-text scores."""

    collection_name: str
    point_id: PointId
    chunk: ChunkT
    dense_score: float | None
    fulltext_score: float
    fulltext_explanation: RerankExplanation


@dataclass(frozen=True, slots=True)
class _Candidate(Generic[ChunkT]):
    point_id: PointId
    chunk: ChunkT
    rerank_text: str
    dense_score: float | None


def render_dense_fields(fields: Sequence[DenseField]) -> str:
    """Render named dense fields in the collection-wide fixed format."""
    return "\n".join(f"{field.name}: {field.text}" for field in fields)


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as error:
        raise RuntimeError(
            "Qdrant support is optional; install rakuhoku[qdrant]"
        ) from error
    return models


class HybridQdrantDB(Generic[ChunkT]):
    """Standard async API for one dense+sparse Qdrant collection."""

    def __init__(
        self,
        *,
        client: Any,
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
        client: Any,
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
        models = _qdrant_models()
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
        client: Any,
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

    def _collection_metadata(self) -> dict[str, Any]:
        config = self.config
        return {
            COLLECTION_METADATA_KEY: {
                "schema_version": SCHEMA_VERSION,
                "dense_model_id": config.dense_model_id,
                "sparse_algorithm_id": config.sparse_algorithm_id,
                "vectorization_schema_id": config.vectorization_schema_id,
                "dense_vector_name": config.dense_vector_name,
                "sparse_vector_name": config.sparse_vector_name,
            }
        }

    def _validate_collection(self, info: Any) -> None:
        models = _qdrant_models()
        vectors: Any = info.config.params.vectors
        sparse_vectors: Any = info.config.params.sparse_vectors
        if not isinstance(vectors, dict):
            raise ValueError("collection does not use named dense vectors")
        dense_vectors = cast(dict[str, Any], vectors)
        dense = dense_vectors.get(self.config.dense_vector_name)
        if dense is None:
            raise ValueError("configured dense vector is missing from collection")
        if dense.size != self.config.dense_dimension:
            raise ValueError("collection dense vector dimension does not match")
        if dense.distance != models.Distance.COSINE:
            raise ValueError("collection dense vector must use cosine distance")
        if not isinstance(sparse_vectors, dict):
            raise ValueError("collection has no named sparse vectors")
        named_sparse_vectors = cast(dict[str, Any], sparse_vectors)
        sparse = named_sparse_vectors.get(self.config.sparse_vector_name)
        if sparse is None:
            raise ValueError("configured sparse vector is missing from collection")
        if sparse.modifier != models.Modifier.IDF:
            raise ValueError("collection sparse vector must use Modifier.IDF")
        metadata = cast(dict[str, Any], info.config.metadata or {})
        if metadata.get(COLLECTION_METADATA_KEY) != self._collection_metadata()[
            COLLECTION_METADATA_KEY
        ]:
            raise ValueError("collection rakuhoku metadata does not match")

    @staticmethod
    def _chunk_payload(
        chunk: ChunkModel,
        *,
        dense_text: str,
        sparse_fields: Sequence[SparseField],
        rerank_text: str,
    ) -> dict[str, Any]:
        payload = chunk.model_dump(mode="json")
        if PAYLOAD_METADATA_KEY in payload:
            raise ValueError(
                f"chunk payload uses reserved key {PAYLOAD_METADATA_KEY!r}"
            )
        payload[PAYLOAD_METADATA_KEY] = {
            "dense_text": dense_text,
            "sparse_fields": [
                {"text": field.text, "weight": field.weight}
                for field in sparse_fields
            ],
            "rerank_text": rerank_text,
            "vectorization_schema_id": chunk.vectorization_schema_id,
        }
        return payload

    async def upsert_chunks(
        self, points: Sequence[ChunkPoint[ChunkT]], *, wait: bool = True
    ) -> None:
        if not points:
            return
        models = _qdrant_models()
        chunks = [point.chunk for point in points]
        for chunk in chunks:
            if not isinstance(chunk, self.chunk_type):
                raise TypeError(
                    f"expected {self.chunk_type.__name__}, got {type(chunk).__name__}"
                )
            if chunk.vectorization_schema_id != self.config.vectorization_schema_id:
                raise ValueError("chunk vectorization schema does not match collection")

        dense_texts = [render_dense_fields(chunk.dense_fields()) for chunk in chunks]
        sparse_fields = [tuple(chunk.sparse_fields()) for chunk in chunks]
        rerank_texts = [chunk.rerank_text() for chunk in chunks]
        dense_vectors = await asyncio.to_thread(
            self.vectorizer.dense_encoder.encode_documents, dense_texts
        )
        if dense_vectors.shape != (len(chunks), self.config.dense_dimension):
            raise RuntimeError("dense encoder returned an unexpected vector shape")
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
                    self.config.sparse_vector_name: models.SparseVector(
                        indices=list(sparse_vectors[index].indices),
                        values=list(sparse_vectors[index].values),
                    ),
                },
                payload=self._chunk_payload(
                    point.chunk,
                    dense_text=dense_texts[index],
                    sparse_fields=sparse_fields[index],
                    rerank_text=rerank_texts[index],
                ),
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
        models = _qdrant_models()
        await self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=wait,
        )

    def _validate_prepared_query(self, query: PreparedQuery) -> None:
        if query.dense_model_id != self.config.dense_model_id:
            raise ValueError("query dense model does not match collection")
        if query.sparse_algorithm_id != self.config.sparse_algorithm_id:
            raise ValueError("query sparse algorithm does not match collection")
        if query.dense_vector.shape != (self.config.dense_dimension,):
            raise ValueError("query dense vector dimension does not match collection")

    def _candidate_from_point(
        self,
        point: Any,
        *,
        dense_score: float | None,
    ) -> _Candidate[ChunkT]:
        if point.payload is None:
            raise ValueError(f"Qdrant point {point.id!r} has no payload")
        payload = dict(cast(dict[str, Any], point.payload))
        library_payload_value = payload.pop(PAYLOAD_METADATA_KEY, None)
        library_payload = cast(dict[str, Any], library_payload_value)
        if not isinstance(library_payload_value, dict):
            raise ValueError(
                f"Qdrant point {point.id!r} has no rakuhoku payload metadata"
            )
        if (
            library_payload.get("vectorization_schema_id")
            != self.config.vectorization_schema_id
        ):
            raise ValueError(f"Qdrant point {point.id!r} has an incompatible schema")
        rerank_text = library_payload.get("rerank_text")
        if not isinstance(rerank_text, str):
            raise ValueError(f"Qdrant point {point.id!r} has no rerank text")
        chunk = self.chunk_type.model_validate(payload)
        return _Candidate(
            point_id=cast(PointId, point.id),
            chunk=chunk,
            rerank_text=rerank_text,
            dense_score=dense_score,
        )

    async def search_by_vectors(
        self,
        query: PreparedQuery,
        *,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        query_filter: Any | None = None,
    ) -> list[ScoredChunk[ChunkT]]:
        """Search one DB, union candidates, and add a lexical full-text score."""
        if dense_limit < 0 or sparse_limit < 0:
            raise ValueError("search limits must be non-negative")
        if dense_limit == 0 and sparse_limit == 0:
            return []
        self._validate_prepared_query(query)
        models = _qdrant_models()

        async def dense_search() -> list[Any]:
            if dense_limit == 0:
                return []
            response = await self.client.query_points(
                collection_name=self.config.collection_name,
                query=query.dense_vector.tolist(),
                using=self.config.dense_vector_name,
                query_filter=query_filter,
                limit=dense_limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(response.points)

        async def sparse_search() -> list[Any]:
            if sparse_limit == 0 or not query.sparse_vector.indices:
                return []
            response = await self.client.query_points(
                collection_name=self.config.collection_name,
                query=models.SparseVector(
                    indices=list(query.sparse_vector.indices),
                    values=list(query.sparse_vector.values),
                ),
                using=self.config.sparse_vector_name,
                query_filter=query_filter,
                limit=sparse_limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(response.points)

        dense_points, sparse_points = await asyncio.gather(
            dense_search(), sparse_search()
        )
        candidates: dict[PointId, _Candidate[ChunkT]] = {}
        for point in dense_points:
            point_id = cast(PointId, point.id)
            candidates[point_id] = self._candidate_from_point(
                point,
                dense_score=float(point.score),
            )
        for point in sparse_points:
            point_id = cast(PointId, point.id)
            if point_id not in candidates:
                candidates[point_id] = self._candidate_from_point(
                    point,
                    dense_score=None,
                )

        query_tokens_results = await asyncio.to_thread(
            self.reranker.rerank,
            query.text,
            [candidate.rerank_text for candidate in candidates.values()],
        )
        candidate_list = list(candidates.values())
        results = [
            ScoredChunk(
                collection_name=self.config.collection_name,
                point_id=(candidate := candidate_list[item.original_index]).point_id,
                chunk=candidate.chunk,
                dense_score=candidate.dense_score,
                fulltext_score=item.score,
                fulltext_explanation=item.explanation,
            )
            for item in query_tokens_results
        ]
        return results

    async def search(
        self,
        query_text: str,
        *,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        query_filter: Any | None = None,
    ) -> list[ScoredChunk[ChunkT]]:
        """Standard one-DB search: encode, retrieve, union, and rerank."""
        prepared = await asyncio.to_thread(self.vectorizer.encode, query_text)
        return await self.search_by_vectors(
            prepared,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            query_filter=query_filter,
        )

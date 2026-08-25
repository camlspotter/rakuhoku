from __future__ import annotations

from qdrant_client.models import SparseVector as QdrantSparseVector

from .types import SparseVector


def to_qdrant_sparse_vector(vector: SparseVector) -> QdrantSparseVector:
    return QdrantSparseVector(
        indices=list(vector.indices), values=list(vector.values)
    )


def from_qdrant_sparse_vector(vector: QdrantSparseVector) -> SparseVector:
    return SparseVector(indices=list(vector.indices), values=list(vector.values))

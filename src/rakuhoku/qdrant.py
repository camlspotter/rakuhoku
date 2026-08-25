from __future__ import annotations

from typing import Any

from .types import SparseVector


def to_qdrant_sparse_vector(vector: SparseVector) -> Any:
    """Convert lazily so importing the core package never requires Qdrant."""
    try:
        from qdrant_client.models import SparseVector as QdrantSparseVector
    except ImportError as error:
        raise RuntimeError(
            "Qdrant support is optional; install rakuhoku[qdrant]"
        ) from error
    return QdrantSparseVector(
        indices=list(vector.indices), values=list(vector.values)
    )


def from_qdrant_sparse_vector(vector: Any) -> SparseVector:
    return SparseVector(indices=list(vector.indices), values=list(vector.values))

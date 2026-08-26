from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel

from .types import DenseField, SparseField


def render_dense_fields(fields: Sequence[DenseField]) -> str:
    """Render named dense fields in the collection-wide fixed format."""
    return "\n".join(f"{field.name}: {field.text}" for field in fields)


class ChunkModel(BaseModel, ABC):
    """Client-defined, validated chunk with a fixed vectorization contract.

    Applications define their payload fields as ordinary Pydantic fields and
    implement the three methods below.  The schema identifier belongs to the
    vectorization behavior, not merely to the JSON payload shape; applications
    must change it whenever any of these methods changes semantically.
    """

    vectorization_schema_id: ClassVar[str]

    @abstractmethod
    def dense_fields(self) -> Sequence[DenseField]:
        """Return ordered, named fields used to construct dense input."""

    def dense_text(self) -> str:
        """Return the exact text passed to the dense document encoder."""
        return render_dense_fields(self.dense_fields())

    @abstractmethod
    def sparse_fields(self) -> Sequence[SparseField]:
        """Return ordered, independently weighted sparse input fields."""

    @abstractmethod
    def rerank_text(self) -> str:
        """Return the actual chunk text compared with a query by a reranker."""

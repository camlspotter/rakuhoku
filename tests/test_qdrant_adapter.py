from __future__ import annotations

from rakuhoku.qdrant import (
    from_qdrant_sparse_vector,
    to_qdrant_sparse_vector,
)
from rakuhoku.types import SparseVector


def test_qdrant_round_trip() -> None:
    source = SparseVector(
        indices=[0, 1, (1 << 32) - 1], values=[0.25, 0.5, 2.0]
    )
    assert from_qdrant_sparse_vector(to_qdrant_sparse_vector(source)) == source

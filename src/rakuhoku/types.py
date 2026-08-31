from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdrant_client import models


@dataclass(frozen=True, slots=True)
class DenseField:
    """A named text field used to construct dense-vector input."""

    name: str
    text: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dense field name must not be empty")


@dataclass(frozen=True, slots=True)
class SparseField:
    """An independently bounded text field and its application-defined weight."""

    text: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("field weight must be non-negative")


FeatureKind = Literal["token", "synonym", "noun_phrase", "char_2gram", "char_3gram"]


@dataclass(frozen=True, slots=True)
class MorphemeExplanation:
    """One Sudachi morpheme exposed by the diagnostic API."""

    field_index: int
    surface: str
    normalized: str
    pos: str
    synonym_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SparseFeature:
    """One pre-TF feature exposed by the diagnostic API."""

    name: str
    kind: FeatureKind
    text: str
    source_text: str
    weight: float
    index: int


@dataclass(frozen=True, slots=True)
class SparseExplanation:
    algorithm_id: str
    dimensions: int
    sudachipy_version: str
    dictionary_version: str
    morphemes: list[MorphemeExplanation]
    features: list[SparseFeature]
    vector: models.SparseVector

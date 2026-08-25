from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .sudachi import Morpheme, SudachiTokenizer
from .types import (
    FeatureKind,
    MorphemeExplanation,
    SparseExplanation,
    SparseFeature,
    SparseField,
    SparseVector,
)


ALGORITHM_ID = "sudachi-sparse-v1"
DEFAULT_DIMENSIONS = 1 << 32
SUPPORTED_SUDACHIPY_VERSION = "0.6.10"
SUPPORTED_DICTIONARY_VERSION = "20260116"


def char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3)) -> list[str]:
    output: list[str] = []
    for size in sizes:
        if size <= 0:
            raise ValueError("n-gram sizes must be positive")
        output.extend(
            text[offset : offset + size]
            for offset in range(len(text) - size + 1)
        )
    return output


@dataclass(frozen=True, slots=True)
class _RawFeature:
    kind: FeatureKind
    text: str
    source_text: str
    weight: float

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.text}"


class SudachiSparseEncoder:
    """Deterministic namespaced Japanese sparse encoder."""

    excluded_pos = frozenset(
        {"空白", "記号", "補助記号", "助詞", "感動詞"}
    )
    feature_weights: dict[FeatureKind, float] = {
        "token": 1.0,
        "synonym": 1.0,
        "noun_phrase": 1.0,
        "char_2gram": 0.05,
        "char_3gram": 0.15,
    }

    def __init__(
        self,
        *,
        dimensions: int = DEFAULT_DIMENSIONS,
        tokenizer: SudachiTokenizer | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if dimensions != DEFAULT_DIMENSIONS:
            raise ValueError(
                f"dimensions is fixed at {DEFAULT_DIMENSIONS} for {ALGORITHM_ID}"
            )
        self.dimensions = dimensions
        self.tokenizer = tokenizer or SudachiTokenizer()
        if self.tokenizer.sudachipy_version != SUPPORTED_SUDACHIPY_VERSION:
            raise RuntimeError(
                "unsupported SudachiPy version: "
                f"{self.tokenizer.sudachipy_version}; "
                f"expected {SUPPORTED_SUDACHIPY_VERSION}"
            )
        if self.tokenizer.dictionary_version != SUPPORTED_DICTIONARY_VERSION:
            raise RuntimeError(
                "unsupported Sudachi dictionary version: "
                f"{self.tokenizer.dictionary_version}; expected {SUPPORTED_DICTIONARY_VERSION}"
            )

    @property
    def algorithm_id(self) -> str:
        return ALGORITHM_ID

    @property
    def dictionary_version(self) -> str:
        return self.tokenizer.dictionary_version

    @property
    def sudachipy_version(self) -> str:
        return self.tokenizer.sudachipy_version

    def hash_feature(self, name: str) -> int:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big")

    def _noun_phrases(self, morphemes: list[Morpheme]) -> list[list[Morpheme]]:
        phrases: list[list[Morpheme]] = []
        current: list[Morpheme] = []
        saw_noun = False
        saw_suffix = False

        def flush() -> None:
            nonlocal current, saw_noun, saw_suffix
            if saw_noun:
                phrases.append(current)
            current = []
            saw_noun = False
            saw_suffix = False

        for morpheme in morphemes:
            if morpheme.pos == "接頭辞":
                if saw_noun or saw_suffix:
                    flush()
                current.append(morpheme)
            elif morpheme.pos == "名詞":
                if saw_suffix:
                    flush()
                current.append(morpheme)
                saw_noun = True
            elif morpheme.pos == "接尾辞":
                if saw_noun:
                    current.append(morpheme)
                    saw_suffix = True
                else:
                    flush()
            else:
                flush()
        flush()
        return phrases

    def _feature(
        self, kind: FeatureKind, text: str, source_text: str, field_weight: float
    ) -> _RawFeature:
        return _RawFeature(
            kind=kind,
            text=text,
            source_text=source_text,
            weight=self.feature_weights[kind] * field_weight,
        )

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return unicodedata.normalize("NFKC", text).casefold()

    def _extract_field(
        self, field: SparseField
    ) -> tuple[list[_RawFeature], list[Morpheme]]:
        if not field.text or field.weight == 0:
            return [], []
        morphemes = self.tokenizer.tokenize(field.text)
        output: list[_RawFeature] = []
        for morpheme in morphemes:
            if morpheme.pos in self.excluded_pos:
                continue
            output.append(
                self._feature(
                    "token",
                    morpheme.normalized,
                    morpheme.surface,
                    field.weight,
                )
            )
            for synonym_id in morpheme.synonym_ids:
                output.append(
                    self._feature(
                        "synonym", str(synonym_id), morpheme.surface, field.weight
                    )
                )

        for phrase in self._noun_phrases(morphemes):
            source_text = "".join(morpheme.surface for morpheme in phrase)
            phrase_text = self._normalize_phrase(source_text)
            if len(phrase) >= 2:
                output.append(
                    self._feature(
                        "noun_phrase", phrase_text, source_text, field.weight
                    )
                )
            for gram in char_ngrams(phrase_text):
                kind = "char_2gram" if len(gram) == 2 else "char_3gram"
                output.append(self._feature(kind, gram, source_text, field.weight))
        return output, morphemes

    def _fields(self, value: str | Iterable[SparseField]) -> list[SparseField]:
        if isinstance(value, str):
            return [SparseField(value)]
        return list(value)

    def _vector(self, raw_features: list[_RawFeature]) -> SparseVector:
        totals_by_name: defaultdict[str, float] = defaultdict(float)
        for feature in raw_features:
            totals_by_name[feature.name] += feature.weight

        index_totals: defaultdict[int, float] = defaultdict(float)
        for name, weight in totals_by_name.items():
            index_totals[self.hash_feature(name)] += math.log1p(weight)

        sorted_items = sorted(index_totals.items())
        return SparseVector(
            indices=[index for index, _ in sorted_items],
            values=[weight for _, weight in sorted_items],
        )

    def explain(
        self, value: str | Iterable[SparseField]
    ) -> SparseExplanation:
        raw_features: list[_RawFeature] = []
        morphemes: list[MorphemeExplanation] = []
        for field_index, field in enumerate(self._fields(value)):
            field_features, field_morphemes = self._extract_field(field)
            raw_features.extend(field_features)
            morphemes.extend(
                MorphemeExplanation(
                    field_index=field_index,
                    surface=morpheme.surface,
                    normalized=morpheme.normalized,
                    pos=morpheme.pos,
                    synonym_ids=morpheme.synonym_ids,
                )
                for morpheme in field_morphemes
            )
        features: list[SparseFeature] = []
        for feature in raw_features:
            index = self.hash_feature(feature.name)
            features.append(
                SparseFeature(
                    name=feature.name,
                    kind=feature.kind,
                    text=feature.text,
                    source_text=feature.source_text,
                    weight=feature.weight,
                    index=index,
                )
            )
        return SparseExplanation(
            algorithm_id=self.algorithm_id,
            dimensions=self.dimensions,
            sudachipy_version=self.sudachipy_version,
            dictionary_version=self.dictionary_version,
            morphemes=morphemes,
            features=features,
            vector=self._vector(raw_features),
        )

    def encode(self, value: str | Iterable[SparseField]) -> SparseVector:
        raw_features: list[_RawFeature] = []
        for field in self._fields(value):
            field_features, _ = self._extract_field(field)
            raw_features.extend(field_features)
        return self._vector(raw_features)

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.encode(text) for text in texts]

    def encode_queries(self, texts: list[str]) -> list[SparseVector]:
        return [self.encode(text) for text in texts]

    def encode_texts_for_insertion(self, texts: list[str]) -> list[SparseVector]:
        return self.encode_documents(texts)

    def encode_texts_for_query(self, texts: list[str]) -> list[SparseVector]:
        return self.encode_queries(texts)

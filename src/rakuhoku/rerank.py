from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Literal, Sequence

from .sudachi import Morpheme, SudachiTokenizer


MatchKind = Literal["exact", "synonym", "ngram"]


def _assert_fulltext_score(score: float) -> float:
    assert math.isfinite(score), f"fulltext_score must be finite, got {score!r}"
    assert 0.0 <= score <= 1.0, f"fulltext_score must be in [0, 1], got {score!r}"
    return score


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Weights and thresholds for the local lexical reranker."""

    coverage_weight: float = 0.6
    order_weight: float = 0.2
    proximity_weight: float = 0.2
    synonym_similarity: float = 0.8
    ngram_similarity_scale: float = 0.6
    ngram_threshold: float = 0.4
    ngram_min_length: int = 3
    gap_penalty: float = 0.1
    boundary_penalty: float = 0.5

    def __post_init__(self) -> None:
        numeric_values = (
            self.coverage_weight,
            self.order_weight,
            self.proximity_weight,
            self.synonym_similarity,
            self.ngram_similarity_scale,
            self.ngram_threshold,
            self.gap_penalty,
            self.boundary_penalty,
        )
        if any(value < 0 for value in numeric_values):
            raise ValueError("rerank parameters must be non-negative")
        if self.coverage_weight + self.order_weight + self.proximity_weight <= 0:
            raise ValueError("at least one score weight must be positive")
        if (
            self.synonym_similarity > 1
            or self.ngram_similarity_scale > 1
            or self.ngram_threshold > 1
        ):
            raise ValueError("similarities and thresholds must not exceed 1")
        if self.ngram_min_length < 1:
            raise ValueError("ngram_min_length must be positive")


@dataclass(frozen=True, slots=True)
class TokenMatch:
    query_index: int
    chunk_index: int
    query_token: str
    chunk_token: str
    kind: MatchKind
    similarity: float


@dataclass(frozen=True, slots=True)
class RerankExplanation:
    score: float
    coverage_score: float
    order_score: float
    proximity_score: float
    query_tokens: tuple[str, ...]
    chunk_tokens: tuple[str, ...]
    coverage_matches: tuple[TokenMatch, ...]
    ordered_matches: tuple[TokenMatch, ...]
    gap_count: int
    boundary_cost: float


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    original_index: int
    text: str
    score: float
    explanation: RerankExplanation


@dataclass(frozen=True, slots=True)
class _Token:
    surface: str
    normalized: str
    synonym_ids: frozenset[int]
    boundary_before: float


@dataclass(frozen=True, slots=True)
class _Match:
    kind: MatchKind
    similarity: float


@dataclass(frozen=True, slots=True)
class _AlignmentNode:
    query_index: int
    chunk_index: int
    match: _Match
    previous: _AlignmentNode | None


@dataclass(frozen=True, slots=True)
class _AlignmentState:
    similarity: float = 0.0
    count: int = 0
    cost: float = 0.0
    last_chunk_index: int = -1
    node: _AlignmentNode | None = None

    @property
    def key(self) -> tuple[float, int, float, int]:
        return (
            self.similarity,
            self.count,
            -self.cost,
            self.last_chunk_index,
        )


class SudachiLexicalReranker:
    """Explainable local reranking against the original chunk text."""

    excluded_pos = frozenset({"空白", "記号", "補助記号"})

    def __init__(
        self,
        *,
        config: RerankConfig | None = None,
        tokenizer: SudachiTokenizer | None = None,
    ) -> None:
        self.config = config or RerankConfig()
        self.tokenizer = tokenizer or SudachiTokenizer()

    @staticmethod
    def _boundary_weight(morpheme: Morpheme) -> float:
        surface = morpheme.surface
        newline_cost = float(surface.count("\n"))
        if morpheme.pos == "空白":
            return newline_cost
        if any(character in "。！？!?" for character in surface):
            return newline_cost + 1.0
        if morpheme.pos in {"記号", "補助記号"}:
            return newline_cost + 0.25
        return newline_cost

    def _tokens(self, text: str) -> list[_Token]:
        output: list[_Token] = []
        boundary_before = 0.0
        for morpheme in self.tokenizer.tokenize(text):
            if morpheme.pos in self.excluded_pos:
                boundary_before += self._boundary_weight(morpheme)
                continue
            output.append(
                _Token(
                    surface=morpheme.surface,
                    normalized=morpheme.normalized,
                    synonym_ids=frozenset(morpheme.synonym_ids),
                    boundary_before=boundary_before,
                )
            )
            boundary_before = 0.0
        return output

    @staticmethod
    def _comparison_text(text: str) -> str:
        return unicodedata.normalize("NFKC", text).casefold()

    @classmethod
    def _character_ngrams(cls, text: str) -> frozenset[str]:
        normalized = cls._comparison_text(text)
        return frozenset(
            normalized[offset : offset + size]
            for size in (1, 2, 3)
            for offset in range(len(normalized) - size + 1)
        )

    @classmethod
    def _ngram_similarity(cls, left: str, right: str) -> float:
        left_grams = cls._character_ngrams(left)
        right_grams = cls._character_ngrams(right)
        if not left_grams or not right_grams:
            return 0.0
        return 2.0 * len(left_grams & right_grams) / (
            len(left_grams) + len(right_grams)
        )

    def _match(self, query: _Token, chunk: _Token) -> _Match | None:
        if query.normalized == chunk.normalized:
            return _Match("exact", 1.0)
        if query.synonym_ids & chunk.synonym_ids:
            return _Match("synonym", self.config.synonym_similarity)

        query_text = self._comparison_text(query.normalized)
        chunk_text = self._comparison_text(chunk.normalized)
        if min(len(query_text), len(chunk_text)) < self.config.ngram_min_length:
            return None
        similarity = self._ngram_similarity(query_text, chunk_text)
        if similarity < self.config.ngram_threshold:
            return None
        return _Match("ngram", similarity * self.config.ngram_similarity_scale)

    def _all_matches(
        self, query_tokens: list[_Token], chunk_tokens: list[_Token]
    ) -> list[list[_Match | None]]:
        return [
            [self._match(query, chunk) for chunk in chunk_tokens]
            for query in query_tokens
        ]

    @staticmethod
    def _public_match(
        query_index: int,
        chunk_index: int,
        query_tokens: list[_Token],
        chunk_tokens: list[_Token],
        match: _Match,
    ) -> TokenMatch:
        return TokenMatch(
            query_index=query_index,
            chunk_index=chunk_index,
            query_token=query_tokens[query_index].normalized,
            chunk_token=chunk_tokens[chunk_index].normalized,
            kind=match.kind,
            similarity=match.similarity,
        )

    def _coverage_matches(
        self,
        query_tokens: list[_Token],
        chunk_tokens: list[_Token],
        matches: list[list[_Match | None]],
    ) -> tuple[TokenMatch, ...]:
        candidates = [
            (match.similarity, query_index, chunk_index, match)
            for query_index, row in enumerate(matches)
            for chunk_index, match in enumerate(row)
            if match is not None
        ]
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_query: set[int] = set()
        used_chunk: set[int] = set()
        selected: list[TokenMatch] = []
        for _, query_index, chunk_index, match in candidates:
            if query_index in used_query or chunk_index in used_chunk:
                continue
            used_query.add(query_index)
            used_chunk.add(chunk_index)
            selected.append(
                self._public_match(
                    query_index,
                    chunk_index,
                    query_tokens,
                    chunk_tokens,
                    match,
                )
            )
        return tuple(sorted(selected, key=lambda item: item.query_index))

    def _ordered_matches(
        self,
        query_tokens: list[_Token],
        chunk_tokens: list[_Token],
        matches: list[list[_Match | None]],
    ) -> tuple[TokenMatch, ...]:
        query_count = len(query_tokens)
        chunk_count = len(chunk_tokens)
        boundary_prefix: list[float] = []
        boundary_total = 0.0
        for token in chunk_tokens:
            boundary_total += token.boundary_before
            boundary_prefix.append(boundary_total)

        empty = _AlignmentState()
        states = [
            [empty for _ in range(chunk_count + 1)]
            for _ in range(query_count + 1)
        ]

        for query_index in range(1, query_count + 1):
            for chunk_index in range(1, chunk_count + 1):
                best = states[query_index - 1][chunk_index]
                left = states[query_index][chunk_index - 1]
                if left.key > best.key:
                    best = left
                match = matches[query_index - 1][chunk_index - 1]
                if match is not None:
                    previous = states[query_index - 1][chunk_index - 1]
                    transition_cost = 0.0
                    if previous.last_chunk_index >= 0:
                        gap_count = max(
                            0,
                            chunk_index - previous.last_chunk_index - 2,
                        )
                        boundary_cost = (
                            boundary_prefix[chunk_index - 1]
                            - boundary_prefix[previous.last_chunk_index]
                        )
                        transition_cost = (
                            self.config.gap_penalty * gap_count
                            + self.config.boundary_penalty * boundary_cost
                        )
                    diagonal = _AlignmentState(
                        similarity=previous.similarity + match.similarity,
                        count=previous.count + 1,
                        cost=previous.cost + transition_cost,
                        last_chunk_index=chunk_index - 1,
                        node=_AlignmentNode(
                            query_index=query_index - 1,
                            chunk_index=chunk_index - 1,
                            match=match,
                            previous=previous.node,
                        ),
                    )
                    if diagonal.key > best.key:
                        best = diagonal
                states[query_index][chunk_index] = best

        selected: list[TokenMatch] = []
        node = states[query_count][chunk_count].node
        while node is not None:
            selected.append(
                self._public_match(
                    node.query_index,
                    node.chunk_index,
                    query_tokens,
                    chunk_tokens,
                    node.match,
                )
            )
            node = node.previous
        selected.reverse()
        return tuple(selected)

    @staticmethod
    def _alignment_cost(
        ordered_matches: tuple[TokenMatch, ...], chunk_tokens: list[_Token]
    ) -> tuple[int, float]:
        if len(ordered_matches) < 2:
            return 0, 0.0
        boundary_prefix: list[float] = []
        total = 0.0
        for token in chunk_tokens:
            total += token.boundary_before
            boundary_prefix.append(total)

        gap_count = 0
        boundary_cost = 0.0
        for previous, current in zip(ordered_matches, ordered_matches[1:]):
            gap_count += max(0, current.chunk_index - previous.chunk_index - 1)
            boundary_cost += (
                boundary_prefix[current.chunk_index]
                - boundary_prefix[previous.chunk_index]
            )
        return gap_count, boundary_cost

    def _explain_tokens(
        self, query_tokens: list[_Token], chunk_tokens: list[_Token]
    ) -> RerankExplanation:
        if not query_tokens or not chunk_tokens:
            return RerankExplanation(
                score=0.0,
                coverage_score=0.0,
                order_score=0.0,
                proximity_score=0.0,
                query_tokens=tuple(token.normalized for token in query_tokens),
                chunk_tokens=tuple(token.normalized for token in chunk_tokens),
                coverage_matches=(),
                ordered_matches=(),
                gap_count=0,
                boundary_cost=0.0,
            )

        matches = self._all_matches(query_tokens, chunk_tokens)
        coverage_matches = self._coverage_matches(
            query_tokens, chunk_tokens, matches
        )
        ordered_matches = self._ordered_matches(query_tokens, chunk_tokens, matches)
        query_count = len(query_tokens)
        coverage_score = sum(
            match.similarity for match in coverage_matches
        ) / query_count
        order_score = sum(match.similarity for match in ordered_matches) / query_count
        gap_count, boundary_cost = self._alignment_cost(
            ordered_matches, chunk_tokens
        )
        proximity_factor = 1.0 / (
            1.0
            + self.config.gap_penalty * gap_count
            + self.config.boundary_penalty * boundary_cost
        )
        proximity_score = order_score * proximity_factor
        weight_total = (
            self.config.coverage_weight
            + self.config.order_weight
            + self.config.proximity_weight
        )
        score = _assert_fulltext_score(
            (
                self.config.coverage_weight * coverage_score
                + self.config.order_weight * order_score
                + self.config.proximity_weight * proximity_score
            )
            / weight_total
        )
        return RerankExplanation(
            score=score,
            coverage_score=coverage_score,
            order_score=order_score,
            proximity_score=proximity_score,
            query_tokens=tuple(token.normalized for token in query_tokens),
            chunk_tokens=tuple(token.normalized for token in chunk_tokens),
            coverage_matches=coverage_matches,
            ordered_matches=ordered_matches,
            gap_count=gap_count,
            boundary_cost=boundary_cost,
        )

    def explain(self, query: str, chunk: str) -> RerankExplanation:
        return self._explain_tokens(self._tokens(query), self._tokens(chunk))

    def score(self, query: str, chunk: str) -> float:
        return self.explain(query, chunk).score

    def rerank(self, query: str, chunks: Sequence[str]) -> list[RerankedChunk]:
        query_tokens = self._tokens(query)
        results = [
            RerankedChunk(
                original_index=index,
                text=chunk,
                score=(explanation := self._explain_tokens(
                    query_tokens, self._tokens(chunk)
                )).score,
                explanation=explanation,
            )
            for index, chunk in enumerate(chunks)
        ]
        return sorted(results, key=lambda result: (-result.score, result.original_index))

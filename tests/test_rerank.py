from __future__ import annotations

import pytest

from rakuhoku import SudachiLexicalReranker
from rakuhoku.rerank import RerankConfig
from rakuhoku.sudachi import Morpheme

from .fakes import FakeTokenizer


def morph(
    surface: str,
    normalized: str | None = None,
    pos: str = "名詞",
    *synonym_ids: int,
) -> Morpheme:
    return Morpheme(
        surface=surface,
        normalized=surface if normalized is None else normalized,
        pos=pos,
        synonym_ids=synonym_ids,
    )


def test_contiguous_ordered_match_beats_distant_and_reversed_matches() -> None:
    reranker = SudachiLexicalReranker()
    query = "利用できない"
    contiguous = reranker.score(query, "この設備は利用できない。")
    distant = reranker.score(query, "この設備を利用することができない。")
    reversed_order = reranker.score(
        query, "できない場合は別の設備を利用する。"
    )
    assert contiguous == pytest.approx(1.0)
    assert contiguous > distant > reversed_order


def test_sentence_and_newline_boundaries_are_soft_penalties() -> None:
    reranker = SudachiLexicalReranker()
    query = "利用できない"
    contiguous = reranker.explain(query, "利用できない")
    divided = reranker.explain(query, "利用。\nできない")
    assert 0 < divided.score < contiguous.score
    assert divided.order_score == pytest.approx(1.0)
    assert divided.boundary_cost > 0


def test_particles_and_auxiliary_verbs_are_kept() -> None:
    explanation = SudachiLexicalReranker().explain(
        "利用してはならない", "利用してはならない"
    )
    assert explanation.query_tokens == (
        "利用",
        "為る",
        "て",
        "は",
        "成る",
        "ない",
    )
    assert explanation.score == pytest.approx(1.0)


def test_single_synonym_match_is_explained_but_has_no_positional_score() -> None:
    reranker = SudachiLexicalReranker()
    exact = reranker.explain("ラボ", "ラボ")
    synonym = reranker.explain("ラボ", "研究所")
    assert exact.score == 0.0
    assert synonym.score == 0.0
    assert exact.available_matches[0].similarity == pytest.approx(1.0)
    assert synonym.available_matches[0].kind == "synonym"
    assert synonym.available_matches[0].similarity == pytest.approx(0.8)


def test_synonym_match_participates_in_order_and_proximity() -> None:
    reranker = SudachiLexicalReranker()
    adjacent = reranker.score("情報ラボ", "情報研究所")
    distant = reranker.score("情報ラボ", "情報とは別に研究所を設置する")
    reversed_order = reranker.score("情報ラボ", "研究所情報")
    assert adjacent > distant > reversed_order


def test_character_ngram_match_recovers_one_character_error() -> None:
    reranker = SudachiLexicalReranker()
    explanation = reranker.explain("システム", "シスデム")
    assert explanation.score == 0.0
    assert explanation.available_matches[0].kind == "ngram"
    assert 0 < explanation.available_matches[0].similarity < 1.0


def test_short_tokens_do_not_use_character_similarity() -> None:
    tokenizer = FakeTokenizer(
        {"query": [morph("情報")], "chunk": [morph("情状")]}
    )
    reranker = SudachiLexicalReranker(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert reranker.score("query", "chunk") == 0


def test_one_chunk_token_cannot_match_repeated_query_tokens_twice() -> None:
    tokenizer = FakeTokenizer(
        {
            "query": [morph("情報"), morph("情報")],
            "chunk": [morph("情報")],
        }
    )
    reranker = SudachiLexicalReranker(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = reranker.explain("query", "chunk")
    assert len(explanation.available_matches) == 1
    assert explanation.score == 0.0


def test_multiple_noun_phrases_are_non_overlapping_position_anchors() -> None:
    explanation = SudachiLexicalReranker().explain(
        "情報・システム研究機構組織運営規則 第３条",
        "情報・システム研究機構組織運営規則\n\n第３条",
    )

    assert explanation.query_tokens == (
        "情報",
        "システム研究機構組織運営規則",
        "第3条",
    )
    assert explanation.order_score == pytest.approx(1.0)
    assert explanation.proximity_score < 1.0


def test_noun_phrase_requires_exact_match_for_positional_evidence() -> None:
    explanation = SudachiLexicalReranker().explain(
        "情報・システム研究機構組織運営規則 第３条",
        "情報・システム研究機構組織運営規則\n\n第１５条",
    )

    assert [match.query_token for match in explanation.available_matches] == [
        "情報",
        "システム研究機構組織運営規則",
    ]
    assert explanation.order_score == 0.0
    assert explanation.proximity_score == 0.0


def test_reversed_complete_evidence_has_no_proximity_score() -> None:
    explanation = SudachiLexicalReranker().explain(
        "情報・システム研究機構組織運営規則 第３条",
        "第３条\n\n情報・システム研究機構組織運営規則",
    )

    assert len(explanation.available_matches) == 3
    assert explanation.order_score < 1.0
    assert explanation.proximity_score == 0.0


def test_order_alignment_prefers_the_closest_equal_quality_occurrence() -> None:
    tokenizer = FakeTokenizer(
        {
            "query": [morph("a"), morph("b")],
            "chunk": [
                morph("a"),
                morph("x"),
                morph("x"),
                morph("b"),
                morph("a"),
                morph("b"),
            ],
        }
    )
    reranker = SudachiLexicalReranker(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = reranker.explain("query", "chunk")
    assert [match.chunk_index for match in explanation.ordered_matches] == [4, 5]
    assert explanation.gap_count == 0


def test_rerank_returns_stable_original_indices_and_explanations() -> None:
    reranker = SudachiLexicalReranker()
    results = reranker.rerank(
        "利用できない",
        [
            "利用できるが持ち帰ることはできない。",
            "利用できない。",
            "関係のない文章。",
        ],
    )
    assert [result.original_index for result in results] == [1, 0, 2]
    assert results[0].score == results[0].explanation.score


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RerankConfig(sparse_weight=-1)

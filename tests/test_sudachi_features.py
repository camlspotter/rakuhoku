from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import pytest

from rakuhoku.sparse import SudachiSparseEncoder
from rakuhoku.sudachi import Morpheme
from rakuhoku.types import SparseExplanation, SparseField

from .fakes import FakeTokenizer


def morph(
    surface: str,
    pos: str = "名詞",
    *synonyms: int,
    normalized: str | None = None,
) -> Morpheme:
    return Morpheme(
        surface,
        surface if normalized is None else normalized,
        pos,
        synonyms,
    )


def names(encoder: SudachiSparseEncoder, text: str) -> set[str]:
    return {feature.name for feature in encoder.explain(text).features}


def test_empty_excluded_pos_and_deictic_pronoun_is_retained() -> None:
    tokenizer = FakeTokenizer(
        {
            "empty": [],
            "excluded": [morph("の", "助詞")],
            "deictic": [morph("これ", "代名詞")],
        }
    )
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert encoder.encode("empty").indices == []
    assert encoder.encode("excluded").indices == []
    assert names(encoder, "deictic") == {"token:これ"}


@pytest.mark.parametrize("pos", ["助動詞", "接続詞", "連体詞", "動詞"])
def test_content_bearing_pos_is_retained(pos: str) -> None:
    tokenizer = FakeTokenizer({"text": [morph("語", pos, 123)]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert names(encoder, "text") == {"token:語", "synonym:123"}


def test_surface_and_each_synonym_are_independent() -> None:
    tokenizer = FakeTokenizer({"研究所": [morph("研究所", "名詞", 860, 6336)]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    feature_names = names(encoder, "研究所")
    assert "token:研究所" in feature_names
    assert "synonym:860" in feature_names
    assert "synonym:6336" in feature_names


def test_strict_noun_phrase_and_suffix_then_noun_split() -> None:
    tokenizer = FakeTokenizer(
        {
            "prefix": [morph("超", "接頭辞")],
            "suffix": [morph("化", "接尾辞")],
            "phrase": [
                morph("超", "接頭辞"),
                morph("情報"),
                morph("システム"),
                morph("化", "接尾辞"),
            ],
            "split": [morph("情報"), morph("化", "接尾辞"), morph("技術")],
            "verb": [morph("教育"), morph("する", "動詞"), morph("活用")],
        }
    )
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert not any(name.startswith("noun_phrase:") for name in names(encoder, "prefix"))
    assert not any(name.startswith("noun_phrase:") for name in names(encoder, "suffix"))
    assert "noun_phrase:超情報システム化" in names(encoder, "phrase")
    assert "noun_phrase:情報化" in names(encoder, "split")
    assert "noun_phrase:情報化技術" not in names(encoder, "split")
    assert "noun_phrase:教育する活用" not in names(encoder, "verb")


def test_noun_phrase_uses_surface_without_pregrouping() -> None:
    tokenizer = FakeTokenizer(
        {
            "text": [
                morph("人間"),
                morph("らし", "接尾辞", normalized="らしい"),
                morph("さ", "接尾辞"),
            ]
        }
    )
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    feature_names = names(encoder, "text")
    assert "noun_phrase:人間らしさ" in feature_names
    assert "noun_phrase:人間らしいさ" not in feature_names


def test_grams_are_generated_from_the_whole_noun_phrase() -> None:
    tokenizer = FakeTokenizer({"text": [morph("情報"), morph("システム")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    features = encoder.explain("text").features
    feature_names = [feature.name for feature in features]
    assert "char_2gram:情報" in feature_names
    assert "char_2gram:報シ" in feature_names
    assert "char_3gram:情報シ" in feature_names
    assert feature_names.count("char_2gram:情報") == 1


def test_single_noun_has_grams_but_no_noun_phrase_feature() -> None:
    tokenizer = FakeTokenizer({"text": [morph("偽情報")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    feature_names = names(encoder, "text")
    assert "token:偽情報" in feature_names
    assert "char_2gram:偽情" in feature_names
    assert "char_2gram:情報" in feature_names
    assert "char_3gram:偽情報" in feature_names
    assert "noun_phrase:偽情報" not in feature_names


def test_phrase_is_nfkc_casefolded_but_token_uses_sudachi_normalized_form() -> None:
    tokenizer = FakeTokenizer(
        {"text": [morph("ＡＩ", normalized="AI"), morph("Ｓｙｓｔｅｍ")]}
    )
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    feature_names = names(encoder, "text")
    assert "token:AI" in feature_names
    assert "token:Ｓｙｓｔｅｍ" in feature_names
    assert "noun_phrase:aisystem" in feature_names
    assert "char_3gram:ais" in feature_names


def test_repeated_grams_are_counted_by_position() -> None:
    tokenizer = FakeTokenizer({"text": [morph("aaaa")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = encoder.explain("text")
    feature_names = [feature.name for feature in explanation.features]
    assert feature_names.count("char_2gram:aa") == 3
    assert feature_names.count("char_3gram:aaa") == 2
    values = dict(zip(explanation.vector.indices, explanation.vector.values))
    assert values[encoder.hash_feature("char_2gram:aa")] == pytest.approx(
        math.log1p(0.15)
    )
    assert values[encoder.hash_feature("char_3gram:aaa")] == pytest.approx(
        math.log1p(0.30)
    )


def test_namespaces_do_not_collapse_same_text() -> None:
    tokenizer = FakeTokenizer({"text": [morph("情報"), morph("情報")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = encoder.explain("text")
    by_name = {feature.name: feature.index for feature in explanation.features}
    assert by_name["token:情報"] != by_name["char_2gram:情報"]
    assert by_name["noun_phrase:情報情報"] != by_name["token:情報"]


def test_blake2b_32_hash_has_no_modulo() -> None:
    encoder = SudachiSparseEncoder(tokenizer=FakeTokenizer({}))  # type: ignore[arg-type]
    expected = int.from_bytes(
        hashlib.blake2b(b"token:test", digest_size=4).digest(),
        "big",
    )
    assert encoder.hash_feature("token:test") == expected


def test_custom_hash_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions is fixed"):
        SudachiSparseEncoder(dimensions=1, tokenizer=FakeTokenizer({}))  # type: ignore[arg-type]


def test_hash_collisions_are_added_and_indices_sorted() -> None:
    class CollidingEncoder(SudachiSparseEncoder):
        def hash_feature(self, name: str) -> int:
            return 0

    tokenizer = FakeTokenizer({"text": [morph("ab", "名詞", 1)]})
    encoder = CollidingEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    vector = encoder.encode("text")
    assert vector.indices == [0]
    assert vector.values == pytest.approx(
        [math.log1p(1.0) + math.log1p(1.0) + math.log1p(0.05)]
    )


def test_query_and_document_vectors_are_identical() -> None:
    tokenizer = FakeTokenizer({"text": [morph("情報")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert encoder.encode_documents(["text"]) == encoder.encode_queries(["text"])


def test_encode_does_not_build_an_explanation() -> None:
    class EncoderWithUnavailableExplanation(SudachiSparseEncoder):
        def explain(
            self, value: str | Iterable[SparseField]
        ) -> SparseExplanation:
            raise AssertionError("encode must not call explain")

    tokenizer = FakeTokenizer({"text": [morph("情報")]})
    encoder = EncoderWithUnavailableExplanation(tokenizer=tokenizer)  # type: ignore[arg-type]
    assert encoder.encode("text").indices


def test_fields_create_boundaries_and_apply_weights() -> None:
    tokenizer = FakeTokenizer({"a": [morph("a")], "b": [morph("b")]})
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = encoder.explain(
        [SparseField("a", weight=2), SparseField("b", weight=0)]
    )
    assert {feature.name for feature in explanation.features} == {"token:a"}
    assert explanation.vector.values == pytest.approx([math.log1p(2)])


def test_explain_includes_morpheme_details_and_field_index() -> None:
    tokenizer = FakeTokenizer(
        {"a": [morph("附属", normalized="付属")], "b": [morph("する", "動詞", 7)]}
    )
    encoder = SudachiSparseEncoder(tokenizer=tokenizer)  # type: ignore[arg-type]
    explanation = encoder.explain([SparseField("a"), SparseField("b")])
    assert [morpheme.field_index for morpheme in explanation.morphemes] == [0, 1]
    assert explanation.morphemes[0].surface == "附属"
    assert explanation.morphemes[0].normalized == "付属"
    assert explanation.morphemes[1].pos == "動詞"
    assert explanation.morphemes[1].synonym_ids == (7,)

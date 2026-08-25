from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rakuhoku import (
    SUPPORTED_DICTIONARY_VERSION,
    SUPPORTED_SUDACHIPY_VERSION,
    SudachiSparseEncoder,
)


def test_dictionary_version_and_normalization() -> None:
    encoder = SudachiSparseEncoder()
    explanation = encoder.explain("附属")
    assert encoder.dictionary_version == SUPPORTED_DICTIONARY_VERSION
    assert encoder.sudachipy_version == SUPPORTED_SUDACHIPY_VERSION
    assert "token:付属" in {feature.name for feature in explanation.features}
    assert explanation.morphemes[0].surface == "附属"
    assert explanation.morphemes[0].normalized == "付属"


def test_shared_tokenizer_can_be_called_concurrently() -> None:
    encoder = SudachiSparseEncoder()
    values = ["生成AIの教育活用", "研究所", "第3条"] * 8
    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(encoder.encode, values))
    assert len(vectors) == len(values)
    assert all(vector.indices == sorted(vector.indices) for vector in vectors)


def test_middle_dot_is_not_removed_before_tokenization() -> None:
    encoder = SudachiSparseEncoder()
    names = {feature.name for feature in encoder.explain("情報・システム").features}
    assert "noun_phrase:情報システム" not in names


def test_whitespace_is_a_noun_phrase_boundary() -> None:
    encoder = SudachiSparseEncoder()
    names = {feature.name for feature in encoder.explain("研究所 ラボ").features}
    assert "noun_phrase:研究所ラボ" not in names


def test_surface_based_noun_phrase_preserves_natural_suffixes() -> None:
    encoder = SudachiSparseEncoder()
    names = {feature.name for feature in encoder.explain("人間らしさ").features}
    assert "noun_phrase:人間らしさ" in names
    assert "noun_phrase:人間らしいさ" not in names

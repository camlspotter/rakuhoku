from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_vectorizers.dense import SentenceTransformerEncoder


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert batch_size == 4
        assert convert_to_numpy
        assert normalize_embeddings
        self.calls.append(sentences)
        return np.array(
            [[len(sentence), sum(sentence.encode())] for sentence in sentences],
            dtype=np.float32,
        )


def make_encoder(tmp_path: Path, model: FakeModel) -> SentenceTransformerEncoder:
    def load(_name: str, _device: str, _local_only: bool) -> FakeModel:
        return model

    return SentenceTransformerEncoder(
        model_name="fake-model",
        insertion_prefix="doc: ",
        query_prefix="query: ",
        cache_path=tmp_path / "cache.sqlite3",
        batch_size=4,
        device="cpu",
        model_loader=load,
    )


def test_model_is_lazy_and_document_cache_deduplicates_batch(tmp_path: Path) -> None:
    model = FakeModel()
    encoder = make_encoder(tmp_path, model)
    assert model.calls == []

    first = encoder.encode_documents(["same", "same", "different"])
    assert model.calls == [["doc: same", "doc: different"]]
    assert np.array_equal(first[0], first[1])

    second = encoder.encode_documents(["same", "different"])
    assert len(model.calls) == 1
    assert np.array_equal(first[[0, 2]], second)


def test_query_uses_prefix_without_cache(tmp_path: Path) -> None:
    model = FakeModel()
    encoder = make_encoder(tmp_path, model)
    encoder.encode_queries(["x"])
    encoder.encode_queries(["x"])
    assert model.calls == [["query: x"], ["query: x"]]


def test_cache_key_includes_model_and_prefixed_text(tmp_path: Path) -> None:
    encoder = make_encoder(tmp_path, FakeModel())
    assert encoder.cache_key("doc: text") == (
        "1ac32f24c248c0eeca26be133fa89962c512b57024cfb2230b004950275a7e0d"
    )

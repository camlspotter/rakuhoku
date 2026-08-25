from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .cache import SQLiteEmbeddingCache


logger = logging.getLogger(__name__)
FloatMatrix = NDArray[np.float32]


class SentenceModel(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> object: ...


ModelLoader = Callable[[str, str, bool], SentenceModel]


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_sentence_transformer(
    model_name: str, device: str, local_files_only: bool
) -> SentenceModel:
    from sentence_transformers import SentenceTransformer

    return cast(
        SentenceModel,
        SentenceTransformer(
            model_name, device=device, local_files_only=local_files_only
        ),
    )


class SentenceTransformerEncoder:
    """Lazy dense encoder with an insertion-only SQLite cache."""

    def __init__(
        self,
        *,
        model_name: str,
        insertion_prefix: str,
        query_prefix: str,
        cache_path: Path,
        batch_size: int = 128,
        device: str | None = None,
        model_loader: ModelLoader = _load_sentence_transformer,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.insertion_prefix = insertion_prefix
        self.query_prefix = query_prefix
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.device = device or select_device()
        self._model_loader = model_loader
        self._model: SentenceModel | None = None
        self._dimension: int | None = None
        self._cache: SQLiteEmbeddingCache | None = None

    @property
    def model(self) -> SentenceModel:
        if self._model is None:
            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            try:
                self._model = self._model_loader(self.model_name, self.device, True)
            except Exception:
                self._model = self._model_loader(self.model_name, self.device, False)
            dimension = self._model.get_sentence_embedding_dimension()
            if not dimension:
                raise RuntimeError(
                    f"model {self.model_name!r} did not report an embedding dimension"
                )
            self._dimension = dimension
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        assert self._dimension is not None
        return self._dimension

    @property
    def dim(self) -> int:
        """Compatibility alias used by the source applications."""
        return self.dimension

    def _get_cache(self) -> SQLiteEmbeddingCache:
        if self._cache is None:
            try:
                self._cache = SQLiteEmbeddingCache(self.cache_path, self.dimension)
            except Exception as error:
                raise RuntimeError(
                    f"failed to initialize embedding cache at {self.cache_path}"
                ) from error
        return self._cache

    def cache_key(self, prefixed_text: str) -> str:
        raw = f"{self.model_name}\n{prefixed_text}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _model_encode(self, inputs: list[str]) -> FloatMatrix:
        encoded = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(inputs) >= 100,
            normalize_embeddings=True,
        )
        result = np.asarray(encoded, dtype=np.float32)
        expected = (len(inputs), self.dimension)
        if result.shape != expected:
            raise RuntimeError(
                f"model returned shape {result.shape}, expected {expected}"
            )
        return result

    def _empty(self) -> FloatMatrix:
        return np.empty((0, self.dimension), dtype=np.float32)

    def _encode_without_cache(self, texts: list[str], prefix: str) -> FloatMatrix:
        if not texts:
            return self._empty()
        return self._model_encode([prefix + text for text in texts])

    def _encode_with_cache(self, texts: list[str], prefix: str) -> FloatMatrix:
        if not texts:
            return self._empty()
        cache = self._get_cache()
        inputs = [prefix + text for text in texts]
        keys = [self.cache_key(text) for text in inputs]
        try:
            cached = cache.get_many(keys)
        except Exception as error:
            raise RuntimeError(
                f"failed to read embedding cache at {self.cache_path}"
            ) from error

        output: list[NDArray[np.float32] | None] = [None] * len(texts)
        missing_keys: list[str] = []
        missing_inputs: list[str] = []
        positions: dict[str, list[int]] = {}
        for position, (key, input_text) in enumerate(zip(keys, inputs, strict=True)):
            if key in cached:
                output[position] = cached[key]
            else:
                if key not in positions:
                    positions[key] = []
                    missing_keys.append(key)
                    missing_inputs.append(input_text)
                positions[key].append(position)

        if missing_inputs:
            new_vectors = self._model_encode(missing_inputs)
            to_store: dict[str, NDArray[np.float32]] = {}
            for offset, key in enumerate(missing_keys):
                vector = new_vectors[offset]
                to_store[key] = vector
                for position in positions[key]:
                    output[position] = vector
            try:
                cache.set_many(to_store)
            except Exception as error:
                raise RuntimeError(
                    f"failed to write embedding cache at {self.cache_path}"
                ) from error

        if any(vector is None for vector in output):
            raise AssertionError("internal error: an embedding result was not assigned")
        return np.stack([vector for vector in output if vector is not None]).astype(
            np.float32, copy=False
        )

    def encode_documents(self, texts: list[str]) -> FloatMatrix:
        return self._encode_with_cache(texts, self.insertion_prefix)

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        return self._encode_without_cache(texts, self.query_prefix)

    def encode_texts_for_insertion(self, texts: list[str]) -> FloatMatrix:
        return self.encode_documents(texts)

    def encode_texts_for_query(self, texts: list[str]) -> FloatMatrix:
        return self.encode_queries(texts)

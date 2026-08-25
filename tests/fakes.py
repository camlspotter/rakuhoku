from __future__ import annotations

import threading

from rag_vectorizers.sudachi import Morpheme


class FakeTokenizer:
    dictionary_version = "20260116"
    sudachipy_version = "0.6.10"

    def __init__(self, mapping: dict[str, list[Morpheme]]) -> None:
        self.mapping = mapping
        self.calls = 0
        self._lock = threading.Lock()

    def tokenize(self, text: str) -> list[Morpheme]:
        with self._lock:
            self.calls += 1
        return list(self.mapping.get(text, []))

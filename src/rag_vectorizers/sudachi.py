from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True, slots=True)
class Morpheme:
    surface: str
    normalized: str
    pos: str
    synonym_ids: tuple[int, ...]


class SudachiTokenizer:
    """Shared SplitMode.C tokenizer protected by a lock."""

    def __init__(self) -> None:
        try:
            from sudachipy import dictionary, tokenizer

            self._tokenizer = dictionary.Dictionary().create()
            self._split_mode = tokenizer.Tokenizer.SplitMode.C
        except Exception as error:
            raise RuntimeError(
                "Sudachi is required; install sudachipy and sudachidict-core"
            ) from error
        self._lock = threading.Lock()

    @property
    def dictionary_version(self) -> str:
        return version("sudachidict-core")

    @property
    def sudachipy_version(self) -> str:
        return version("sudachipy")

    def tokenize(self, text: str) -> list[Morpheme]:
        with self._lock:
            raw_morphemes = list(self._tokenizer.tokenize(text, self._split_mode))
        output: list[Morpheme] = []
        for raw in raw_morphemes:
            surface = raw.surface()
            normalized = raw.normalized_form()
            if not surface and not normalized:
                continue
            part_of_speech = raw.part_of_speech()
            output.append(
                Morpheme(
                    surface=surface,
                    normalized=normalized,
                    pos=part_of_speech[0] if part_of_speech else "UNK",
                    synonym_ids=tuple(raw.synonym_group_ids()),
                )
            )
        return output

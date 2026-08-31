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


def noun_phrase_spans(morphemes: list[Morpheme]) -> list[tuple[int, int]]:
    """Return half-open spans for prefix/noun/suffix noun phrases."""
    spans: list[tuple[int, int]] = []
    start = 0
    saw_noun = False
    saw_suffix = False

    def flush(end: int) -> None:
        nonlocal start, saw_noun, saw_suffix
        if saw_noun:
            spans.append((start, end))
        start = end
        saw_noun = False
        saw_suffix = False

    for index, morpheme in enumerate(morphemes):
        if morpheme.pos == "接頭辞":
            if saw_noun or saw_suffix:
                flush(index)
            if not saw_noun and not saw_suffix:
                start = index
        elif morpheme.pos == "名詞":
            if saw_suffix:
                flush(index)
            if not saw_noun:
                start = min(start, index)
            saw_noun = True
        elif morpheme.pos == "接尾辞":
            if saw_noun:
                saw_suffix = True
            else:
                flush(index + 1)
        else:
            flush(index)
            start = index + 1
    flush(len(morphemes))
    return spans

from __future__ import annotations

import json
from pathlib import Path

from rakuhoku import SudachiSparseEncoder

from .golden_cases import GOLDEN_CASES


def test_sparse_golden_vectors() -> None:
    path = Path(__file__).parent / "golden" / "sudachi_sparse_v1.json"
    golden = json.loads(path.read_text(encoding="utf-8"))
    encoder = SudachiSparseEncoder()

    assert golden["algorithm_id"] == encoder.algorithm_id
    assert golden["sudachipy_version"] == encoder.sudachipy_version
    assert golden["dictionary_version"] == encoder.dictionary_version
    assert golden["dimensions"] == encoder.dimensions
    assert list(golden["cases"]) == GOLDEN_CASES

    actual_cases: dict[str, object] = {}
    for text in GOLDEN_CASES:
        explanation = encoder.explain(text)
        actual_cases[text] = {
            "features": [
                {
                    "name": feature.name,
                    "weight": feature.weight,
                    "index": feature.index,
                }
                for feature in explanation.features
            ],
            "vector": {
                "indices": explanation.vector.indices,
                "values": explanation.vector.values,
            },
        }
    assert actual_cases == golden["cases"]

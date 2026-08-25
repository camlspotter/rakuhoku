"""Regenerate golden sparse vectors. Run explicitly; tests never rewrite fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_vectorizers import SudachiSparseEncoder

from .golden_cases import GOLDEN_CASES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="confirm replacement of the checked-in golden fixture",
    )
    args = parser.parse_args()
    if not args.update:
        parser.error("fixture generation requires --update")

    encoder = SudachiSparseEncoder()
    cases: dict[str, object] = {}
    for text in GOLDEN_CASES:
        explanation = encoder.explain(text)
        cases[text] = {
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
    output = {
        "algorithm_id": encoder.algorithm_id,
        "sudachipy_version": encoder.sudachipy_version,
        "dictionary_version": encoder.dictionary_version,
        "dimensions": encoder.dimensions,
        "cases": cases,
    }
    path = Path(__file__).parent / "golden" / "sudachi_sparse_v1.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

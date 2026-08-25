from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from rakuhoku.cache import SQLiteEmbeddingCache


def test_cache_round_trip_and_dimension_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    with SQLiteEmbeddingCache(path, dimension=2) as cache:
        cache.set_many({"valid": np.array([1, 2], dtype=np.float32)})
        assert cache.get_many(["valid"])["valid"].tolist() == [1.0, 2.0]

    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT OR REPLACE INTO cache (k, vec) VALUES (?, ?)",
        ("wrong", np.array([1], dtype=np.float32).tobytes()),
    )
    connection.commit()
    connection.close()

    with SQLiteEmbeddingCache(path, dimension=2) as cache:
        assert "wrong" not in cache.get_many(["wrong"])


def test_cache_handles_more_than_sqlite_parameter_limit(tmp_path: Path) -> None:
    with SQLiteEmbeddingCache(tmp_path / "cache.sqlite3", dimension=1) as cache:
        values = {
            f"key-{index}": np.array([index], dtype=np.float32)
            for index in range(1000)
        }
        cache.set_many(values)
        assert len(cache.get_many(list(values))) == 1000

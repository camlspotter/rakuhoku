from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatVector = NDArray[np.float32]


class SQLiteEmbeddingCache:
    """Persistent float32 embedding cache safe for concurrent callers."""

    def __init__(self, db_path: Path, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.db_path = db_path
        self.dimension = dimension
        self._lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(db_path), timeout=30.0, check_same_thread=False
        )
        with self._lock:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )
            self._connection.commit()

    def get_many(self, keys: list[str]) -> dict[str, FloatVector]:
        if not keys:
            return {}
        output: dict[str, FloatVector] = {}
        with self._lock:
            for offset in range(0, len(keys), 900):
                chunk = keys[offset : offset + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT k, vec FROM cache WHERE k IN ({placeholders})", chunk
                ).fetchall()
                for key, blob in rows:
                    vector = np.frombuffer(blob, dtype=np.float32)
                    if vector.size == self.dimension:
                        output[str(key)] = vector.copy()
        return output

    def set_many(self, items: dict[str, FloatVector]) -> None:
        if not items:
            return
        rows = [
            (key, np.asarray(vector, dtype=np.float32).tobytes())
            for key, vector in items.items()
        ]
        with self._lock:
            self._connection.executemany(
                "INSERT OR REPLACE INTO cache (k, vec) VALUES (?, ?)", rows
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteEmbeddingCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

# rag-vectorizers

`rag-vectorizers`は、再現可能な密ベクトルと日本語の疎ベクトルを生成するための
小さなPythonパッケージである。Qdrantのコレクション、ペイロード、クエリ展開、
score fusion、アプリケーション固有の入力テキスト構築は、このパッケージの責務に
含めない。sparse検索後の候補文字列を比較するローカル字句rerankerは、独立した
補助機能として提供する。

> [!IMPORTANT]
> 2026-08-18に`SudachiSparseEncoder`の仕様選択とローカル実装への反映を完了した。
> 確定した選択と根拠は[docs/sparse-spec.md](docs/sparse-spec.md)に記録している。
> アプリケーションの移行とDBの再作成はまだ行っていない。

## 密ベクトル

```python
from pathlib import Path

from rag_vectorizers import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder(
    model_name="cl-nagoya/ruri-v3-130m",
    insertion_prefix="検索文書: ",
    query_prefix="検索クエリ: ",
    cache_path=Path("var/embedding-cache.sqlite3"),
    batch_size=128,
)

document_vectors = encoder.encode_documents(["検索対象の文書"])
query_vectors = encoder.encode_queries(["検索語"])
```

モデルは遅延読み込みする。文書ベクトルにはfloat32形式のSQLiteキャッシュを使用し、
クエリベクトルにはキャッシュを使用しない。どちらも正規化された埋め込みを生成する。

## 疎ベクトル

```python
from rag_vectorizers import SparseField, SudachiSparseEncoder

encoder = SudachiSparseEncoder()
vector = encoder.encode("生成AIの教育活用")
weighted = encoder.encode(
    [SparseField("タイトル", weight=2.0), SparseField("本文")]
)
explanation = encoder.explain("研究所")
```

中核となる`SparseVector`型はQdrantに依存しない。Qdrantへの変換が必要な環境だけ、
オプションのアダプターをインストールする。

```console
uv add 'rag-vectorizers[qdrant]'
```

```python
from rag_vectorizers.qdrant import to_qdrant_sparse_vector

qdrant_vector = to_qdrant_sparse_vector(vector)
```

このベクトルを使用するQdrantコレクションでは`Modifier.IDF`を設定する。ベクトル名と
コレクションの作成・更新・削除は、アプリケーション側の責務とする。

## ローカルreranking

```python
from rag_vectorizers import SudachiLexicalReranker

reranker = SudachiLexicalReranker()
results = reranker.rerank(
    "利用できない",
    [
        "利用できるが、持ち帰ることはできない。",
        "この設備は利用できない。",
    ],
)

for result in results:
    print(result.original_index, result.score, result.text)
```

候補の実際のchunk文字列に対して、query tokenのcoverage、順序、距離、句読点・改行境界を
評価する。正規形で一致しないtokenは、Sudachi synonym、文字n-gramの順に弱い一致として
扱う。詳細は[ローカルreranker仕様](docs/reranking.md)を参照する。

## Chunkモデル

アプリケーション固有のchunkは、`ChunkModel`を継承したPydanticモデルとして定義できる。
`dense_fields()`、`sparse_fields()`、`rerank_text()`により、構造化payloadとvector化・
reranking用テキストの作り方を分離する。ライブラリはpayload全体の共通スキーマを規定しない。

```python
from collections.abc import Sequence
from typing import ClassVar

from rag_vectorizers import ChunkModel, DenseField, SparseField


class TextChunk(ChunkModel):
    vectorization_schema_id: ClassVar[str] = "text-chunk-v1"
    title: str
    text: str

    def dense_fields(self) -> Sequence[DenseField]:
        return (DenseField("title", self.title), DenseField("text", self.text))

    def sparse_fields(self) -> Sequence[SparseField]:
        return (SparseField(self.title), SparseField(self.text))

    def rerank_text(self) -> str:
        return self.text
```

詳細は[Chunkモデルのインターフェース](docs/chunk-model.md)を参照する。

## 互換性

確定した疎ベクトル仕様のアルゴリズムIDは`sudachi-sparse-v1`であり、
`sudachipy==0.6.10`と`sudachidict-core==20260116`を厳密に使用する。ただし、
実装とgolden vectorもこの仕様へ更新済みである。アルゴリズムID、辞書バージョン、
ハッシュ範囲、特徴規則、重みの異なるベクトルを同じコレクションへ混在させては
ならない。詳細は
[確定した疎ベクトル仕様](docs/sparse-spec.md)と
[移行・切り戻し手順](docs/migration.md)を参照する。

## 開発

```console
uv sync --extra qdrant
uv run pytest
uv run pyright
```

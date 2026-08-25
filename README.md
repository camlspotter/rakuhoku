# rakuhoku

`rakuhoku`は、再現可能な密ベクトルと日本語の疎ベクトルを生成するための
小さなPythonパッケージである。dense・sparse vector生成に加え、単一Qdrant
collectionへの登録と標準検索APIを提供する。標準検索はdense・sparse候補をpoint IDで
統合し、保存済みchunk文字列から全文検索scoreを付ける。dense scoreと全文検索scoreの
最終的な合成はアプリケーションが決める。

> [!IMPORTANT]
> 2026-08-18に`SudachiSparseEncoder`の仕様選択とローカル実装への反映を完了した。
> 確定した選択と根拠は[docs/sparse-spec.md](docs/sparse-spec.md)に記録している。
> アプリケーションの移行とDBの再作成はまだ行っていない。

## 密ベクトル

```python
from pathlib import Path

from rakuhoku import SentenceTransformerEncoder

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
from rakuhoku import SparseField, SudachiSparseEncoder

encoder = SudachiSparseEncoder()
vector = encoder.encode("生成AIの教育活用")
weighted = encoder.encode(
    [SparseField("タイトル", weight=2.0), SparseField("本文")]
)
explanation = encoder.explain("研究所")
```

中核となる`SparseVector`型はQdrantの型から分離されており、標準で提供する
アダプターを介してQdrantの疎ベクトルへ変換する。

```console
uv add rakuhoku
```

```python
from rakuhoku.qdrant import to_qdrant_sparse_vector

qdrant_vector = to_qdrant_sparse_vector(vector)
```

低水準APIで既存コードへ組み込む場合、このベクトルを使用するQdrantコレクションでは
`Modifier.IDF`を設定する。新規コードでは、次の統合APIがこの設定を管理する。

## Qdrant標準検索API

一つのDBを検索するときは、原則として`HybridQdrantDB.search()`を使用する。

```python
from qdrant_client import AsyncQdrantClient
from rakuhoku import (
    HybridDBConfig,
    HybridQdrantDB,
    QueryVectorizer,
    SudachiLexicalReranker,
)

client = AsyncQdrantClient(url="http://localhost:6333")
vectorizer = QueryVectorizer(
    dense_encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
)
config = HybridDBConfig(
    collection_name="documents",
    dense_vector_name="dense",
    sparse_vector_name="sparse",
    dense_dimension=vectorizer.dense_dimension,
    dense_model_id=vectorizer.dense_model_id,
    sparse_algorithm_id=vectorizer.sparse_algorithm_id,
    vectorization_schema_id=DocumentChunk.vectorization_schema_id,
)

db = await HybridQdrantDB.open(
    client=client,
    config=config,
    chunk_type=DocumentChunk,
    vectorizer=vectorizer,
    reranker=SudachiLexicalReranker(),
)
results = await db.search("利用できない", dense_limit=50, sparse_limit=50)

for result in results:
    print(result.chunk, result.dense_score, result.fulltext_score)
```

sparse検索のQdrant scoreは候補取得にだけ使い、結果には残さない。dense由来、sparse由来の
全候補に同じ全文検索rerankerを適用する。複数DBでquery vectorを再利用する場合は、
`QueryVectorizer.encode()`で一度だけ`PreparedQuery`を作り、各DBの
`search_by_vectors()`へ渡す。作成・登録を含む完全なAPIは
[Qdrant統合API](docs/qdrant.md)を参照する。

## ローカルreranking

```python
from rakuhoku import SudachiLexicalReranker

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

from rakuhoku import ChunkModel, DenseField, SparseField


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
uv sync
uv run pytest
uv run pyright
```

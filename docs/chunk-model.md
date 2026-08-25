# Chunkモデルのインターフェース

アプリケーションは、chunkのpayloadを`ChunkModel`を継承したPydanticモデルとして
定義する。ライブラリはpayload全体の共通スキーマを規定しない。アプリケーションが
各フィールドの型、fallback、文字列化、vector化対象を決定する。

```python
from collections.abc import Sequence
from typing import ClassVar

from rag_vectorizers import ChunkModel, DenseField, SparseField


class RegulationChunk(ChunkModel):
    vectorization_schema_id: ClassVar[str] = "regulation-v1"

    source_url: str
    title: str
    article: str | None
    text: str

    def dense_fields(self) -> Sequence[DenseField]:
        return (
            DenseField("title", self.title),
            DenseField("article", self.article or ""),
            DenseField("text", self.text),
        )

    def sparse_fields(self) -> Sequence[SparseField]:
        return (
            SparseField(self.title, weight=2.0),
            SparseField(self.article or ""),
            SparseField(self.text),
        )

    def rerank_text(self) -> str:
        return self.text
```

## 必須メソッド

### `dense_fields()`

dense入力に使うフィールドを、`DenseField(name, text)`の順序付きシーケンスとして返す。
フィールド名を含む実際のdenseテキストへのレンダリング規則は、Qdrant統合APIとともに
別途定義する。

### `sparse_fields()`

sparse入力に使う値を`SparseField(text, weight)`として返す。`SparseField`には意図的に
フィールド名がなく、各要素は独立した境界としてSudachi sparse encoderへ渡される。

### `rerank_text()`

ローカルrerankerがqueryと比較する実際のchunk文字列を返す。これは`sparse_fields()`の
単純な連結からは導出しない。タイトルなどをsparse検索に利用しながら、rerankingでは
本文だけを比較するユースケースがあるためである。

## Vectorization schema ID

`vectorization_schema_id`はPydanticのpayloadフィールドではなくクラス変数である。同じ
Qdrantコレクションへ登録するchunkは同じIDを使用する。3つのメソッドのフィールド選択、
順序、ラベル、fallback、重み、rerank対象を変更した場合はIDも変更する。

IDの保存と既存コレクションを開く際の検証は、今後のQdrant統合APIで実装する。

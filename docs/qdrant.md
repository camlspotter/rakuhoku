# Qdrant統合API

`HybridQdrantDB`は、一つのdense+sparse Qdrant collectionに対する標準APIである。
I/Oメソッドは非同期だが、クラス名には`Async`を付けない。Qdrant clientの生成、認証、
closeは呼び出し側が管理する。

## 検索結果

`search()`は次を一続きで行う。

1. queryのdense vectorとsparse vectorを一度ずつ生成する。
2. dense検索とsparse検索を並行して実行する。
3. 同一collection内の結果をQdrant point IDで統合する。
4. 全候補の保存済み`rerank_text`をローカルrerankerで評価する。
5. 全文検索scoreの降順で`list[ScoredChunk]`を返す。

`ScoredChunk`の主要フィールドは次のとおりである。

| フィールド | 内容 |
|---|---|
| `collection_name` | 検索したcollection |
| `point_id` | Qdrant point ID |
| `chunk` | 指定した`ChunkModel`型へ復元したpayload |
| `dense_score` | dense候補ならQdrant cosine score、sparseのみなら`None` |
| `fulltext_score` | ローカル字句rerankerによる0〜1の比較値 |
| `fulltext_explanation` | 通常は`None`。指定時のみcoverage、順序、距離などの内訳 |

sparseのQdrant scoreは候補生成にだけ使い、返却しない。二つのscoreを一つに合成したり、
異なるDBの結果を統合・重複排除したりする処理はクライアント側の責務である。

説明が必要な場合は明示的に指定する。この引数は`search()`と
`search_by_vectors()`の両方で使用できる。

```python
results = await db.search(
    "利用できない",
    with_fulltext_explanation=True,
)
```

## 作成と既存collectionのopen

```python
config = HybridDBConfig(
    collection_name="documents",
    dense_vector_name="dense",
    sparse_vector_name="sparse",
    dense_dimension=vectorizer.dense_dimension,
    dense_model_id=vectorizer.dense_model_id,
    sparse_algorithm_id=vectorizer.sparse_algorithm_id,
    vectorization_schema_id=DocumentChunk.vectorization_schema_id,
)

db = await HybridQdrantDB.create(
    client=client,
    config=config,
    chunk_type=DocumentChunk,
    vectorizer=vectorizer,
)
```

`create()`はdense vectorをCosine、sparse vectorを`Modifier.IDF`で作る。同名collectionが
既にあれば上書きせずエラーにする。`open()`は存在確認に加え、vector名、dense次元、
Cosine、IDF、dense model ID、sparse algorithm ID、chunkのvectorization schema IDを
検証する。互換性metadataを持たない既存collectionは開かない。

## chunkの登録

```python
await db.upsert_chunks(
    [
        ChunkPoint(id=point_id, chunk=chunk),
    ]
)
```

ライブラリは`dense_text()`の値から文書dense vectorを生成する。既定の`dense_text()`は
`dense_fields()`を`name: text`の改行区切りで描画する。`sparse_fields()`はフィールド名を
加えず、独立した境界と重みを保ったままsparse encoderへ渡す。

元のPydantic payloadはトップレベルへ保存する。予約key`_rakuhoku`には、実際に
vector化へ渡したdense文字列とsparse fields、`rerank_text()`の値、schema IDを保存する。
検索時に外部ファイルや別のpayloadフィールドからrerank文字列を再構築しない。

## query vectorの再利用

```python
prepared = vectorizer.encode("利用できない")

results_a = await db_a.search_by_vectors(prepared)
results_b = await db_b.search_by_vectors(prepared)
```

`PreparedQuery`はquery文字列、1次元`numpy.float32` dense vector、`SparseVector`、dense
model ID、sparse algorithm IDを持つ。`search_by_vectors()`はIDと次元の互換性を検証する。
一つのDBだけを検索する通常経路では、同じ処理をまとめた`search()`を使用する。

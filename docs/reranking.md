# ローカル字句reranker仕様

## 1. 目的

`SudachiLexicalReranker`は、sparse検索で取得した候補の実際のchunk文字列とqueryを比較し、
候補内の順位を付け直す。sparse vectorの生成規則や`sudachi-sparse-v1`のalgorithm IDには
影響しない。

rerankerは取得済み候補だけを並べ替える。候補集合に入らなかった文書は復活できないため、
sparse検索ではrerank件数に十分な候補数を取得する必要がある。

## 2. API

```python
from rag_vectorizers import SudachiLexicalReranker

reranker = SudachiLexicalReranker()

score = reranker.score(query, chunk)
explanation = reranker.explain(query, chunk)
results = reranker.rerank(query, chunks)
```

- `score()`は`0.0`から`1.0`の比較用scoreを返す。確率ではない。
- `explain()`はscore内訳、token対応、gap数、境界costを返す。
- `rerank()`はscore降順、同点時は元の候補順で返す。各結果に`original_index`を保持する。

## 3. token化

queryとchunkを`sudachipy==0.6.10`、`sudachidict-core==20260116`、`SplitMode.C`で
tokenizeする。

空白、記号、補助記号は照合tokenから除外する。助詞、助動詞、動詞などは、否定や条件表現を
評価するため保持する。照合文字列にはSudachiの`normalized_form()`を使用する。

```text
利用してはならない
-> 利用 / 為る / て / は / 成る / ない
```

## 4. token一致

同じquery tokenとchunk tokenについて、最初に成立した方式を採用する。

| 一致方式 | 条件 | similarity |
|---|---|---:|
| 正規形一致 | `normalized_form()`が同じ | `1.0` |
| synonym一致 | Sudachi synonym group IDを一つ以上共有 | `0.8` |
| 文字n-gram一致 | 後述する類似度が閾値以上 | `0.6 × n-gram類似度` |

文字n-gram一致は、正規形一致とsynonym一致のどちらも成立しなかった場合だけ使用する。
NFKCとcase folding後の1/2/3-gram集合についてDice係数を計算する。両tokenが3文字以上で、
係数が`0.4`以上の場合だけ一致とする。短い語の曖昧な部分一致を避けながら、一文字程度の
OCR誤りや書き損じを弱い一致として拾うためである。

## 5. score

scoreは次の三要素から作る。

| 要素 | 内容 | weight |
|---|---|---:|
| coverage | query tokenのうちchunk内のtokenで説明できた割合 | `0.6` |
| order | queryと同じ順序で対応できたtokenの割合 | `0.2` |
| proximity | order scoreへ距離と境界の減点を適用 | `0.2` |

```text
score = 0.6 × coverage + 0.2 × order + 0.2 × proximity
```

chunk長では正規化しない。長いchunkそのものを不利にせず、queryをどれだけ説明できるかを
主軸にする。

同じchunk tokenを複数のquery tokenへ重複使用しない。orderはtokenの最大重み付き共通部分列で
計算するため、順序逆転があると低下する。

## 6. 距離と境界

順序を保って一致したtoken間に別tokenがある場合、1 tokenごとに`0.1`のgap costを加える。
句読点と改行は一致を禁止せず、soft boundaryとしてcostを加える。

```text
同じ文内で連続       減点なし
同じ順序だが離れている gapによる減点
句点や改行を横断       boundaryによる追加減点
順序が逆転             order scoreが低下
```

proximityは次の係数をorder scoreへ掛ける。

```text
1 / (1 + 0.1 × gap数 + 0.5 × boundary cost)
```

句点や改行を絶対的な境界にしないため、OCRや書き損じによる不自然な分断があっても一致を
残せる。

## 7. 制限

このrerankerは字句的な順序と近接性を評価する。係り受けや否定スコープを完全には解析しない。
たとえば離れた位置にある`利用`、`出来る`、`ない`は減点できるが、すべての文法的な否定関係を
確定できるわけではない。

初期weightと閾値は固定された検索品質ではなく、offline評価の出発点である。実コーパスでは、
rerank前の候補recallを維持したうえで、MRR、NDCG、上位K件の非関連率を比較する。

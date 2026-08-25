# RAG Vectorizers 共通化プロジェクト引き継ぎ

## 1. この文書の目的

`dxsympo` と `rois-rag` に重複している次の実装を、独立した Python
パッケージへ共通化する。

- SentenceTransformer による dense embedding
- SQLite による embedding cache
- Sudachi を使った日本語 sparse vector 生成
- Qdrant の `Modifier.IDF` と組み合わせるための最小限の型・変換処理

新プロジェクトは、アプリケーション固有の検索処理を集約するものではない。
主な責務は「入力テキストから再現可能なdense/sparse vectorを生成するところまで」とする。
例外として、呼び出し側が渡したqueryと候補chunk文字列だけを比較するローカル字句rerankerを
提供する。Qdrantからの候補取得、payload解釈、score fusion、最終cutは引き続き
アプリケーション側の責務とする。

仮のプロジェクト名は `rakuhoku` とする。この名前は確定事項ではない。

## 2. 調査したソースのスナップショット

調査日: 2026-08-18

### dxsympo

- repository: `/Users/jun/dxsympo`
- branch: `fix-search`
- commit: `e2a4a642c793ca7fb483a3334a417399dec47b5c`
- dense: `db/embedding.py`
- sparse feature生成: `db/sparse_sudachi.py`
- BM25 document weight: `db/sparse_sudachi_bm25.py`
- Qdrant query: `db/db_qdrant.py`
- index構築: `tools/tool_build_db.py`
- sparse vector名: `sudachi_bm25`

作業ツリーには `api/public_api.py`、`db/hybrid_search.py` などの未コミット変更が
あるため、新プロジェクトへの移植時にそれらを巻き込まないこと。

### rois-rag

- repository: `/Users/jun/rois-rag`
- branch: `codex/qdrant-idf-bm25`
- commit: `e21549b03a02bb5f400374994e0d18f464457cd2`
- dense: `embedding/embedding.py`
- dense設定: `embedding/config.py`
- sparse feature生成: `db/sparse_japanese.py`
- Qdrant query: `db/db_qdrant.py`
- index構築: `db/vector_store.py`
- sparse vector名: `japanese_sparse`

このブランチは `master` より3コミット先であり、Qdrant IDF化、名詞句規則修正、
同義語feature修正を含む。新プロジェクトでは上記commitの実装を参照すること。

## 3. 共通化の境界

### 新プロジェクトへ移すもの

- dense encoder本体
- insertion/query prefix
- embedding modelのlazy load
- CPU/CUDA/MPSの選択
- embeddingの正規化
- SQLite cache
- cache key生成
- Sudachi tokenizerの生成と排他制御
- morphemeの正規化
- 品詞フィルタ
- 名詞句抽出
- 同義語、文字n-gramなどのsparse feature生成
- featureの重み付け
- feature IDのhash化
- vector index衝突の集約
- vector indexの安定した並べ替え
- アルゴリズムIDと互換性情報
- ライブラリ独自のdense/sparse vector型
- 必要ならQdrant型への薄い変換関数

### 各アプリケーションへ残すもの

- environment variableやJSONからの設定読み込み
- collection名
- Qdrant上のvector名
- Qdrant clientの生成
- collectionの作成・更新・削除
- payload schemaとpayload index
- 文書本文、タイトル、sceneなどの入力テキスト組み立て
- metadata/scene単位のindex構築
- filter、grouped search
- query expansion
- dense/sparseの候補数
- score normalization、rank fusion、結果cut
- `slug` 単位のscene集約
- API schema

`dxsympo` の hybrid search はmetadataとsceneを別経路で検索・集約するため、
共通ライブラリへ移してはならない。

## 4. 推奨パッケージ構造

```text
rakuhoku/
├── pyproject.toml
├── README.md
├── src/
│   └── rakuhoku/
│       ├── __init__.py
│       ├── types.py
│       ├── dense.py
│       ├── cache.py
│       ├── sudachi.py
│       ├── sparse.py
│       └── qdrant.py
└── tests/
    ├── test_dense_cache.py
    ├── test_dense_encoder.py
    ├── test_sudachi_tokenization.py
    ├── test_sudachi_features.py
    ├── test_sparse_golden_vectors.py
    └── test_qdrant_adapter.py
```

`qdrant.py` はoptional dependencyとし、coreのsparse encoderが
`qdrant_client.models.SparseVector` を直接返さない構成を推奨する。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]
```

## 5. dense実装の方針

両リポジトリのdense実装はほぼ同一である。次を共通仕様とする。

- `SentenceTransformer`
- insertion/queryで別prefix
- `normalize_embeddings=True`
- insertionのみSQLite cacheを使用
- queryはcacheを使わない
- cache keyは `sha256(model_name + "\n" + prefixed_text)`
- cache値はfloat32のbytes
- 同一batch内の重複入力を一度だけencode
- cache内のvector次元が異なるレコードはmissとして扱う
- modelはlazy load
- batch sizeは設定可能
- cache pathは呼び出し側が明示する

設定ファイルやenvironment variableをライブラリ内で直接読まないこと。

想定API:

```python
dense = SentenceTransformerEncoder(
    model_name="cl-nagoya/ruri-v3-130m",
    insertion_prefix="検索文書: ",
    query_prefix="検索クエリ: ",
    cache_path=cache_path,
    batch_size=128,
)

document_vectors = dense.encode_documents(texts)
query_vectors = dense.encode_queries(queries)
```

## 6. sparse実装で共通している部分

両実装には次の共通点がある。

- SudachiPyを利用
- `Tokenizer.SplitMode.C`
- `normalized_form().strip().lower()`
- 助詞、助動詞、記号、補助記号、接続詞、連体詞、感動詞を除外
- 指示語の一部を除外
- 表層形featureを保持
- Sudachi同義語group IDをfeature化
- 名詞句をfeature化
- 文字2-gram、3-gramを生成
- Blake2b、digest size 8
- `2^20` 次元へhash
- 同じhash indexへ入った値を加算
- Qdrant collection側で `Modifier.IDF` を利用

ただし、同じ入力から生成されるvectorは互換ではない。代表入力で生成index集合を
比較すると、Jaccard係数は概ね6%から25%だった。

| 入力 | index集合のJaccard係数（概算） |
|---|---:|
| `データダイエット` | 0.06 |
| `情報システム化` | 0.11 |
| `生成AIの教育活用` | 0.10 |
| `研究所 ラボ` | 0.11 |
| `第3条` | 0.25 |

したがって、共通化後のsparse algorithmを導入する際は、両方とも再インデックスが必要。

## 7. 利用者が選択すべきsparse仕様

以下は実装担当者だけで決めず、利用者が検索特性として選択する項目である。
「推奨」は安全性・再現性を優先した初期案であり、検索評価で覆してよい。

> [!NOTE]
> この節は、選択前に提示した比較材料を履歴として残したものである。
> 2026-08-18に全項目の選択を完了した。確定内容は本書の「14. sparse仕様の
> 決定結果」と[詳細な仕様・決定記録](docs/sparse-spec.md)を正とする。

### 決定1: feature namespaceを分離するか

**dxsympo**

- 表層形: `データ`
- 名詞句: `np:データダイエット`
- 2-gram: `c2:デー`
- 3-gram: `c3:データ`
- 同義語: `__syn_860`

**rois-rag**

- 表層形、名詞句、gramは文字列そのものをhashする
- 同じ文字列ならfeature種別を越えて同じindexに集約される
- 同義語だけ専用文字列へ変換する

**選択による影響**

- namespace分離: featureの意味が明確で、偶然の加算を防げる
- 非分離: 同じ文字列の一致を強くできるが、表層形とgramの区別が消える

**推奨:** namespaceを分離する。

- [ ] namespaceを分離する
- [ ] 同じ文字列はfeature種別にかかわらず集約する

### 決定2: 文字n-gramをどこから生成するか

**dxsympo**

- 各名詞token内で2/3-gramを生成
- 複合名詞を結合したときに新しく生まれる境界gramも生成
- token内ですでに生成済みのgramは重複追加しない

**rois-rag**

- 抽出した名詞句全体から2/3-gramを生成
- groupの切り方により生成範囲が変わる

**選択による影響**

- token-local + 境界gram: どのgramを補強しているか説明しやすい
- 名詞句全体: 実装は単純だが、grouping変更でgram集合も変化する

**推奨:** token-local + 名詞句境界gram。

- [ ] token-local + 名詞句境界gram
- [ ] 名詞句全体のgram

### 決定3: 空白と中黒 `・` を語句境界として扱うか

**dxsympo**

- 入力全体を一度にSudachiへ渡す
- 空白だけでは名詞句を必ずしも分断しない
- 中黒に特別規則はない

**rois-rag**

- 最初に `text.split()` し、空白ごとに独立してtokenizeする
- `・` は存在しなかったものとしてgroup内で無視する

**選択による影響**

- 空白で分断: keyword列や別fieldが誤って一つの名詞句になるのを防ぐ
- 空白を保持: OCRや自然文中の不自然な空白に強くなる可能性がある
- 中黒を無視: `情報・システム` を結合できる一方、列挙境界も消える

**推奨:** 呼び出し側からfield境界を明示できるAPIを作り、通常の文中空白は
Sudachiに処理させる。中黒は無条件に消さず、評価対象にする。

- [ ] field境界だけを明示的に分断する
- [ ] すべての空白で分断する

中黒:

- [ ] 通常のtokenとして扱う
- [ ] 常に無視して前後を結合する

### 決定4: 名詞句の範囲

両実装とも基本文法は `接頭辞* + 名詞+ + 接尾辞*` だが、前段のgroupingが異なる。

**dxsympo**

- 全morpheme列へ直接この文法を適用
- 規則外の品詞でflush

**rois-rag**

- 先にgroup化
- 除外品詞でgroupを切る
- 接尾辞、動詞、副詞を追加した直後にgroupを閉じる
- 中黒は無視する

**推奨:** groupingと名詞句文法を分離し、名詞句自体は厳密に
`接頭辞* + 名詞+ + 接尾辞*` とする。動詞・副詞は名詞句へ入れない。

- [ ] 厳密な `接頭辞* + 名詞+ + 接尾辞*`
- [ ] rois-ragのgrouping規則を先に適用する

### 決定5: feature weight

**dxsympo**

- 表層形、同義語、名詞句: 基本 `1.0`
- 2-gram: `0.05`
- 3-gram: `0.15`
- 名詞feature: encode後に `1.15` 倍
- query全体: encode後に `1.2` 倍

**rois-rag**

- 名詞、名詞句、keyword、同義語:
  - 1文字は `0.75`
  - 2文字以上は `1 + 0.15 * log2(length)`
- gram: `length * 0.1`（2-gramは0.2、3-gramは0.3）
- その他の品詞: `0.75`
- query専用倍率なし

**選択による影響**

- dxsympo方式はfeature種別の寄与を直接制御しやすい
- rois-rag方式は長い語を強くするが、長いOCR誤認語も強化する可能性がある
- query全体を一律に定数倍しても単独sparse rankingには影響しない場合があるが、
  raw scoreをdenseと混ぜる場合は影響する

**推奨:** feature種別ごとの明示的な固定weightから開始し、長さboostは評価後に追加する。
query一律倍率はcoreから除外する。

- [ ] 固定weight
- [ ] 文字列長によるweight

query倍率:

- [ ] 使用しない
- [ ] 使用する（倍率: `______`）

### 決定6: TF変換と文書長補正

**dxsympo**

1. featureのboostをtermごとに合計
2. `log1p(tf)`
3. 名詞/query倍率
4. index構築時に別encoderでBM25風のTF飽和・文書長補正
5. Qdrantがcollection IDFを適用

文書長は元token数ではなく、sparse vectorの異なるindex数で近似している。

**rois-rag**

1. feature boostをhash indexごとに合計
2. `log1p(weight)`
3. Qdrantがcollection IDFを適用

追加の文書長補正はない。

**選択による影響**

- dxsympo方式: 長文が大量のfeatureを持つ影響を抑えられる可能性がある
- rois-rag方式: 単純で、Qdrantへ渡した重みを説明しやすい
- dxsympo方式は「BM25」という名前でも、文書長をfeature index数で近似した独自式である

**推奨:** まず `log1p(weight) + Qdrant IDF` を基準とし、文書長補正は独立した
評価可能なpolicyとして追加する。採用する場合はtoken数、feature数、文字数のどれを
文書長とするかも明示する。

- [ ] `log1p(weight) + Qdrant IDF` のみ
- [ ] BM25風のTF飽和・文書長補正を追加

文書長の定義:

- [ ] 異なるfeature index数
- [ ] Sudachi token数
- [ ] 文字数
- [ ] その他: `______`

### 決定7: タイトル強調をcoreへ含めるか

**dxsympo**

- metadata textの構築時にタイトルを一度含める
- sparse encoder自体にtitle引数はない

**rois-rag**

- document encodeに `title` 引数がある
- `text` に含まれているタイトルをさらに追加して強調する

**推奨:** core encoderはtitleを知らない。アプリケーション側がfieldとweightを指定する。
単純な文字列重複より、次のようなfield入力を将来的に許容する方が明確。

```python
SparseField(text=title, weight=2.0)
SparseField(text=body, weight=1.0)
```

- [ ] title強調はアプリケーション側のfield weightで指定する
- [ ] encoderにtitle専用引数を持たせる

### 決定8: 同義語feature

両実装とも現在は次を意図している。

- 元の表層形featureを必ず残す
- synonym group IDごとに独立したfeatureを追加する
- 複数IDを一つの組としてhashしない

例: `研究所=[860, 6336]` と `ラボ=[859, 860]` は、表層形は別indexだが
synonym ID `860` では一致する。

**推奨:** この仕様で統一する。同義語IDの文字列表現はnamespace付きで固定する。

- [ ] 表層形 + synonym IDごとの独立feature
- [ ] 同義語featureを使用しない

### 決定9: hash次元数と衝突規則

**dxsympo**

- 次元数をconstructorで変更可能
- 衝突したindexの値を加算
- 出力をindex順にsort

**rois-rag**

- `2^20` 固定
- 衝突したindexの値を加算
- dict挿入順のまま出力

**推奨:** `2^20` をversion付きの既定値にし、設定変更は許可する。出力はindex順にsortする。

- [ ] `2^20`、安定sort
- [ ] 別の次元数: `______`

### 決定10: Sudachi辞書version

**dxsympo dependency lower bound:** `sudachidict-core>=20241021`

**rois-rag dependency lower bound:** `sudachidict-core>=20260116`

辞書versionが変わると、分割、normalized form、同義語IDが変わり、同じライブラリversionでも
vectorが変化しうる。

**推奨:** 対応する辞書version範囲を狭く固定し、algorithm compatibility IDへ辞書系列を含める。
golden testは実際にsupportする辞書versionで実行する。

- [ ] 辞書versionを厳密に固定する
- [ ] minor範囲を許可し、複数versionでgolden testする

### 決定11: tokenizerの並行利用

**dxsympo:** tokenizer呼び出しをlockで保護する。

**rois-rag:** lockなし。

**推奨:** singleton tokenizerを共有するならlockで保護する。高い並列性が必要なら
thread-local tokenizerを別途benchmarkする。

- [ ] 共有tokenizer + lock
- [ ] thread-local tokenizer

### 決定12: debug featureを公開するか

**dxsympo:** 通常APIはvectorのみ返す。内部token型はprivate。

**rois-rag:** vectorとfeature一覧をtupleで返す。`TOKEN.show()` も持つ。

**推奨:** 通常APIはvectorのみ返し、`explain(text)` という明示的な診断APIで
feature、weight、hash indexを返す。

- [ ] `encode()` と `explain()` を分離する
- [ ] 常にvectorとfeature一覧を返す

## 8. 確定した初期仕様の概要

2026-08-18の検討で、初期仕様を次のように確定した。根拠と追加決定は
「14. sparse仕様の決定結果」および[docs/sparse-spec.md](docs/sparse-spec.md)を参照する。

```text
algorithm_id: sudachi-sparse-v1
split_mode: C
tokenizer: sudachipy==0.6.10
dictionary: sudachidict-core==20260116
hash: blake2b, digest_size=4, big endian, moduloなし
features:
  token: Sudachi normalized_form、namespaced
  synonym: synonym IDごと、namespaced
  noun_phrase: 複数morphemeの句だけ、表層形由来、namespaced
  char_2gram: 名詞句全体の表層形由来、namespaced、weight 0.05
  char_3gram: 名詞句全体の表層形由来、namespaced、weight 0.15
n-grams:
  名詞句全体、出現位置ごとに加算
noun_phrase_grammar:
  pregroupingなしの 接頭辞* + 名詞+ + 接尾辞*
boundaries:
  空白と中黒
surface_normalization:
  NFKC + case folding
excluded_pos:
  空白、記号、補助記号、助詞、感動詞
lexical_stopwords:
  none
weights:
  token/synonym/noun_phrase 1.0、char_2gram 0.05、char_3gram 0.15
tf:
  log1p(weight sum)
document_length_normalization:
  disabled
idf:
  Qdrant Modifier.IDF
query_global_multiplier:
  disabled
output_order:
  sort by index
tokenizer_concurrency:
  shared tokenizer + lock
debug:
  separate explain API
position_and_order:
  application-side reranking
```

## 9. algorithm versionと再インデックス

次のいずれかを変えた場合は、同じalgorithm IDを使い続けてはならない。

- Sudachi辞書の互換性を破る変更
- tokenizer split mode
- 正規化
- 除外品詞
- 名詞句規則
- n-gram生成範囲
- namespace
- synonym IDの表現
- hash関数または次元数
- feature weight
- TF変換
- 文書長補正

新旧vectorを同じcollectionで混在させない。collection名へalgorithm IDを含める。

```text
dxsympo_metadata_ruri-v3-130m_sudachi-v1
dxsympo_scenes_ruri-v3-130m_sudachi-v1
staff_page3_ruri-v3-130m_sudachi-v1
```

algorithm変更時は新collectionを構築し、評価後に参照先を切り替える。

## 10. テスト要件

### 単体テスト

- 空入力
- 空白、記号、補助記号、助詞、感動詞の除外
- 助動詞、接続詞、連体詞など保持対象POS
- 指示表現を語彙stopwordとして除外しないこと
- 空白と中黒による名詞句の分断
- 接頭辞だけ、接尾辞だけでは名詞句を作らない
- `接頭辞* + 名詞+ + 接尾辞*`
- pregroupingを行わないこと
- suffix後にnounが来た場合の分割
- token featureに`normalized_form()`を使うこと
- 名詞句文字列に表層形の連結とNFKC＋case foldingを使うこと
- 単独名詞にはnoun phrase featureを付けず、2/3-gramは作ること
- synonym IDを一つずつfeature化
- `研究所` と `ラボ` が共有IDで一致
- POS filter通過後の全morphemeでsynonymを利用すること
- 名詞句全体から2/3-gramを作ること
- 同じgramを出現位置ごとに加算すること
- 固定weightと`log1p`
- feature namespaceが衝突しない
- Blake2b-32、big endian、剰余なし
- hash衝突時の加算
- index sort
- query/documentで同じfeature IDとweightを使う
- tokenizerの並行呼び出し
- `encode()`と`explain()`の分離

### golden vectorテスト

少なくとも次を固定する。

- `データダイエット`
- `研究所`
- `ラボ`
- `研究所 ラボ`
- `第3条`
- `情報システム化`
- `生成AIの教育活用`
- `日本生協連`
- `情報・システム`
- OCR由来の空白を含む語

golden dataにはfeature名、weight、hash index、最終vectorを保存する。

### 検索評価

両アプリケーションで現行実装と新実装を比較する。

- Recall@K
- MRR
- NDCG@K
- 上位K件の非関連率
- short query
- 固有名詞
- 表記揺れ
- 同義語
- 部分一致
- OCR誤り
- 長文書と短文書の偏り

`dxsympo` ではmetadataとsceneを実際のindex生成関数で組み立てた文字列を使うこと。
raw JSON全体を評価対象にしない。

## 11. 実装・移行順序

1. 新しい独立repositoryを作る。
2. dense encoderとSQLite cacheを移植する。
3. denseの互換テストを両repositoryの入力で通す。
4. `dxsympo` と `rois-rag` を共通denseへ切り替える。
5. sparse仕様を決定し、根拠とともに文書化する。（完了）
6. `sudachi-sparse-v1`とgolden testを確定仕様へ合わせる。（完了）
7. 両コーパスで現行sparseとのoffline評価を行う。
8. 両アプリケーションに新algorithm IDのcollectionを作る。
9. 全文書を再インデックスする。
10. 検索結果を比較し、参照collectionを切り替える。
11. 安定後に各repositoryの旧dense/sparse実装を削除する。

## 12. 配布方法

初期段階ではPyPI公開を前提にしない。Git tagまたはcommitをlockfileで固定する。

```toml
[project]
dependencies = [
    "rakuhoku",
]

[tool.uv.sources]
rakuhoku = {
    git = "https://github.com/camlspotter/rakuhoku.git",
    tag = "v0.1.0",
}
```

ローカルで3repositoryを同時開発するときだけpath sourceへ差し替える。
本番・CIではGit commit/tagを固定し、ローカルディレクトリ構成へ依存させない。

## 13. 完了条件

- dense/cacheの重複実装が両repositoryからなくなる
- sparse仕様の全選択項目が文書化されている
- algorithm IDが公開されている
- golden vectorが安定している
- Qdrant依存なしでcoreテストが実行できる
- Qdrant adapterが別テストになっている
- 両repositoryで型検査と対象テストが通る
- 新collectionへの再インデックスと検索評価が完了している
- 旧collectionへ戻せる切り戻し手順がある
- 旧実装の削除は切り替え確認後に行われる

## 14. sparse仕様の決定結果

決定日: 2026-08-18

以下を`SudachiSparseEncoder`の確定仕様とする。ローカル実装とgolden vectorは
2026-08-18にこの仕様へ更新した。DB作成と他repositoryへの移行は行っていない。

詳細な規則、具体例、境界条件は[docs/sparse-spec.md](docs/sparse-spec.md)に記載する。

### 14.1 featureと入力境界

1. **feature namespaceを分離する。** `token`、`synonym`、`noun_phrase`、
   `char_2gram`、`char_3gram`を別namespaceにする。同じ文字列でも、完全なtoken一致と
   部分一致では検索上の意味が異なり、偶然の加算を避ける必要があるためである。

2. **token featureにはSudachiの`normalized_form()`を使う。** `附属`と`付属`のような
   表記揺れを同じtokenとして検索できる。一方、表層の一致は名詞句由来gramが保持する。

3. **coreはtitle、metadata、sceneなどを知らず、汎用的な`SparseField(text, weight)`を
   受け取る。** これらは`dxsympo`固有の構造であり、`rois-rag`には同じ概念がない。
   用途固有の入力構築と強調をapplication側へ残せば、二重投入も防げる。

4. **vectorizerに渡された空白は単語境界とする。** OCRによる不自然な空白など、
   境界にしたくない空白は利用者が入力前に除去する。全空白の事前削除は
   `machine learning`まで結合する一方、tokenize後の除去では途中で分割された語を
   復元できないためである。

5. **中黒`・`も境界とする。** 無条件に消すと、列挙された別の語まで一つの名詞句として
   接続するためである。

### 14.2 名詞句とcharacter n-gram

6. **pregroupingを行わず、morpheme列へ`接頭辞* + 名詞+ + 接尾辞*`を直接適用する。**
   pregroupingでは`人間 / らし / さ`を`人間らし`と`さ`へ不自然に切断し得る。

7. **名詞句文字列とそのn-gramはmorphemeの表層形から作る。** 正規形の連結では
   `人間 + らしい + さ`から`人間らしいさ`のような不自然な形が生じる。表層形を
   連結後、Unicode NFKCとcase foldingを適用する。

8. **2/3-gramは抽出した名詞句全体から作る。** 名詞句の境界規則が確定したため、
   token内gramと境界gramを別処理するより単純で一貫する。

9. **単独名詞からも2/3-gramを作るが、`noun_phrase` featureは複数morphemeの句だけに
   付ける。** 単独名詞へtokenとnoun phraseを重ねて完全一致を二重強調しないためである。

10. **同じn-gramを出現位置ごとに数える。** 通常の名詞句や対象OCRでは不自然な重複は
    稀であり、正当な反復とOCRノイズをcoreが推測して区別しない。TF増加は後段の
    `log1p`が緩和する。

### 14.3 weight、TF、IDF

11. **固定weightを使う。** token、synonym、noun phraseは`1.0`、2-gramは`0.05`、
    3-gramは`0.15`とする。長い語は既にgram数が多く、珍しい語はIDFも高くなるので、
    文字列長boostを重ねない。偶然一致しやすい2-gramは3-gramより弱くする。

12. **query専用倍率を使わない。** 一律倍率はpure sparse内の順位をほぼ変えず、denseとの
    比率調整ならapplicationのfusion policyとして明示すべきだからである。documentと
    queryには同じfeature規則とweightを用いる。

13. **同一featureのraw weightを合計して`log1p`を適用し、Qdrant側で
    `Modifier.IDF`を使う。BM25風のTF飽和・文書長補正は追加しない。** BM25風の式や
    文書長の定義には複数の選択肢があり、根拠なく一つへ固定できない。必要性が評価で
    示された場合は別algorithmとして検討する。

### 14.4 synonym、品詞、stopword

14. **元tokenを保持し、Sudachiのsynonym group IDを一つずつ独立featureとして追加する。**
    複数IDを一組にすると、一部のgroup IDを共有する語同士が一致できないためである。

15. **synonymはPOS filterを通過した全morphemeで利用する。** 名詞だけに限定せず、辞書が
    持つ形容詞などの言い換えも一貫して利用する。誤一致は検索評価で確認する。

16. **tokenから除外するのは空白、記号、補助記号、助詞、感動詞だけとする。** 助動詞は
    `ない`のような否定、接続詞は規程文書の条件関係、連体詞は`大きな`などの意味を
    持ち得るため保持する。代名詞、動詞、形容詞、形状詞、副詞、名詞、接頭辞、接尾辞も
    保持する。

17. **指示表現を含む語彙stopword listは設けない。** `これ・この・こんな・こういう`などを
    一貫して網羅するには複数morphemeの規則まで必要になり、coreが複雑になる。頻出語は
    IDFへ任せ、実害が確認されてから再検討する。

18. **character n-gramの対象は名詞句だけとする。** 全品詞へ広げるとfeature数と偶然一致が
    大きく増える一方、名詞句の部分一致という目的がぼやけるためである。

### 14.5 hash、version、API

19. **complete namespaced feature stringをBlake2bの32 bit digestでhashする。**
    `digest_size=4`、big endian、剰余なしとする。Qdrantの`uint32`範囲を使い切って衝突を
    抑えられ、狭いhash空間にしてもsparse vectorの保存量は減らないためである。

20. **hash衝突時はvalueを加算し、index昇順で出力する。** 情報を落とさず、同じ入力に
    対して再現可能なvectorにする。

21. **`sudachipy==0.6.10`、`sudachidict-core==20260116`、`SplitMode.C`へ厳密に固定する。**
    辞書変更だけでも分割、品詞、正規形、synonym IDが変わり、vector互換性を失うためである。

22. **encoderごとにtokenizerを一つ共有し、tokenize呼び出しをlockする。** 辞書初期化と
    memory使用を抑えつつ安全性を確保する。性能問題が測定された場合だけthread-local化を
    再検討する。

23. **通常の`encode()`と診断用`explain()`を分離する。** 通常経路に重い説明情報を
    持ち込まず、必要なときだけmorpheme、feature、weight、hash衝突を調査できるようにする。

24. **algorithm IDを`sudachi-sparse-v1`とする。** 辞書、境界、品詞、feature、weight、hash、
    TF規則などvector互換性に関わる変更時は新IDを付け、新collectionへ全件再インデックスする。

### 14.6 語順・否定とreranking

25. **token sequence featureや述語句featureはv1へ追加しない。** `利用できない`と
    `利用できるが持って帰ることはできない`は、位置を持たないsparse featureだけでは
    区別できない。sequenceを増やしても否定の係り先を完全には解決せず、`利用可能`と
    `利用できる`の意味的な対応にもならない。

26. **sparseは高recallの候補取得を担当し、位置、語順、文・節、否定、係り受けは
    事後rerankingへ任せる。** sparse encoderへは組み込まず、元chunk文字列を比較する独立した
    `SudachiLexicalReranker`として提供する。これは取得済み候補だけを並べ替え、sparse
    algorithm IDや既存vectorには影響しない。

# `sudachi-sparse-v1` 仕様・決定記録

この文書は、プロジェクト引き継ぎ文書にあった12項目と、その後の検討で
明らかになった追加項目について、利用者が選択した仕様と根拠を記録する。

> [!IMPORTANT]
> 2026-08-18時点では、この文書が確定仕様であり、ローカル実装とgolden vectorも
> この仕様へ更新済みである。アプリケーションの移行とDBの再作成は行っていない。

## 1. 確定仕様の概要

| 項目 | `sudachi-sparse-v1` の決定 |
|---|---|
| algorithm ID | `sudachi-sparse-v1` |
| feature namespace | `token`、`synonym`、`noun_phrase`、`char_2gram`、`char_3gram`を分離 |
| token文字列 | Sudachiの`normalized_form()` |
| field | 任意個の`SparseField(text, weight)`。title専用概念は持たない |
| 空白 | 利用者が入力した単語境界。vectorizer内で事前除去しない |
| 中黒 `・` | 語句境界。前後を結合しない |
| 名詞句文法 | pregroupingなしで`接頭辞* + 名詞+ + 接尾辞*`を直接適用 |
| 名詞句文字列 | morphemeの表層形を連結し、Unicode NFKCとcase foldingを適用 |
| noun phrase feature | 複数morphemeからなる名詞句だけに追加 |
| character n-gram | 全名詞句の表層由来文字列全体から2/3-gramを生成 |
| n-gram TF | 同じn-gramの出現位置ごとに加算 |
| weights | token/synonym/noun phrase `1.0`、2-gram `0.05`、3-gram `0.15` |
| query倍率 | 使用しない |
| TF変換 | 同じfeatureのweightを合計して`log1p` |
| 文書長補正 | 使用しない |
| IDF | Qdrant collection側の`Modifier.IDF` |
| synonym | POS filter通過後の全morphemeに対して、group IDを一つずつ追加 |
| POS filter | 空白、記号、補助記号、助詞、感動詞を除外 |
| 語彙stopword | 使用しない。指示表現もIDFへ任せる |
| hash | Blake2b、`digest_size=4`（32 bit）、剰余なし |
| hash衝突 | 同じindexのvalueを加算 |
| 出力順 | indexの昇順 |
| Sudachi | `SplitMode.C`、`sudachipy==0.6.10` |
| 辞書 | `sudachidict-core==20260116` |
| concurrency | encoderごとにtokenizerを一つ共有し、tokenize呼び出しをlock |
| debug API | `encode()`と`explain()`を分離 |
| document/query | 同じfeature ID、weight、変換規則を使う |
| 位置・語順 | core vectorには保持しない。候補取得後のrerankingへ任せる |

## 2. feature namespaceを分離する

feature種別を文字列のnamespaceで分離する。

```text
token:情報
noun_phrase:情報システム
char_2gram:情報
char_3gram:情報シ
synonym:860
```

たとえば`偽情報`では、tokenと3-gramが同じ文字列になり得る。

```text
token:偽情報
char_3gram:偽情報
```

namespaceを分離しない場合、完全なtoken一致と部分一致が同じindexへ集約され、
意図せず強調される。また、query中のgram `情報` と、文書中の完全なtoken
`情報`も区別できなくなる。分離することで、完全一致と部分一致を別の根拠として
説明し、feature種別ごとにweightを調整できる。

## 3. token featureはSudachi正規形を使う

token featureには表層形ではなくSudachiの`normalized_form()`を使う。

```text
表層形: 附属
正規形: 付属
feature: token:付属
```

これにより、`附属`と`付属`が同じtoken featureで一致する。表層形の差は、
後述する名詞句由来のn-gramが保持する。同じ表記同士は正規形tokenに加えて
表層由来gramも一致するため、異表記同士より自然に強くなる。

表層形と正規形の両方をtoken featureにする案は採用しない。多くの語では両者が
同じであり、重複規則が必要になるうえ、正規化が発生した語だけfeature数が変わる
ためである。

## 4. fieldは汎用的な`SparseField`とする

core encoderはmetadata、scene、title、bodyなどのアプリケーション固有概念を
持たない。任意個のfieldとweightだけを受け取る。

```python
[
    SparseField(text=title, weight=2.0),
    SparseField(text=body, weight=1.0),
]
```

別field間では名詞句やn-gramを接続しない。同じfeatureが複数fieldに現れた場合は、
field weightを掛けた値を合計する。

この設計により、アプリケーション側は次のような異なる構造を同じAPIへ写像できる。

```text
dxsympo metadata:
  title、speaker、keywords、abstractなど

dxsympo scene:
  scene title、markdown、transcriptなど

rois-rag:
  document title、chunk bodyなど
```

title専用引数は、titleの二重投入やcoreへの用途固有ロジック混入を招くため採用しない。
dense encoderのfield強調はこの決定の対象外である。

## 5. 空白と中黒は境界とする

vectorizerへ渡された空白は、利用者が意図した単語境界として扱う。
OCR由来など、境界にしたくない空白はライブラリ利用者が入力前に除去する。

```text
情報システム   -> 一つの名詞句になり得る
情報 システム  -> 空白で分断
情 報          -> 「情」と「報」に分断
```

tokenize後に空白tokenだけを捨てる方式は採用しない。その方式では、Sudachiが既に
`情 / 空白 / 報`へ分割した後なので、`token:情報`を復元できない。一方、すべての
空白をtokenize前に削除すると、`machine learning`を`machinelearning`へ壊す。
OCR補正はコーパスの事情を知る利用者側の責務とする。

中黒`・`も境界とする。

```text
情報・システム -> 「情報」と「システム」に分断
```

中黒を無条件に消すと、列挙まで一語として接続するためである。

## 6. pregroupingを行わず、名詞句文法を直接適用する

morpheme列を独自のgroupへ分割してから文法を適用する方式は採用しない。

```text
morpheme列 -> 接頭辞* + 名詞+ + 接尾辞*
```

pregroupingは、有効な名詞句を文法適用前に切断し得る。たとえばSudachiは
`人間らしさ`を概ね次のように分割する。

```text
人間  名詞
らし  接尾辞
さ    接尾辞
```

接尾辞を一つ追加した時点でgroupを閉じる規則では、`人間 + らし`と`さ`が
別groupになる。直接文法なら`名詞 + 接尾辞 + 接尾辞`として扱える。

動詞、副詞、助動詞、助詞、連体詞など、文法に含まれないmorphemeは名詞句を
自然にflushする。空白、中黒、記号も前述のとおり境界になる。

## 7. 名詞句とcharacter n-gramは表層形から作る

名詞句の構造判定には品詞を使うが、名詞句featureの文字列には各morphemeの
表層形を使う。正規形を単純連結すると、次のような不自然な語を作るためである。

```text
表層形: 人間 + らし + さ     -> 人間らしさ
正規形: 人間 + らしい + さ   -> 人間らしいさ
```

表層形を連結した後、Unicode NFKCとcase foldingを適用する。

```text
ＡＩシステム -> aiシステム
AIシステム   -> aiシステム
aiシステム   -> aiシステム
```

辞書正規形で活用語尾を置換することはないため、`人間らしさ`の自然な形は保たれる。

## 8. n-gramは名詞句全体から作る

token-local gramと境界gramを別々に管理せず、抽出した名詞句の表層由来文字列全体から
2-gramと3-gramを生成する。

```text
情報 / システム / 化
-> 情報システム化
-> char_2gram:情報、char_2gram:報シ、...
-> char_3gram:情報シ、char_3gram:報シス、...
```

名詞句の範囲、空白、中黒、句読点の規則が既に確定しているため、名詞句全体から
生成する方が単純で一貫する。正規形tokenと表層形境界gramが混在する複雑な実装も
避けられる。

単独名詞もgram生成上は名詞句として扱う。ただし`noun_phrase:` featureは、
複数morphemeから構成された場合だけ追加する。

```text
偽情報
-> token:偽情報
-> char_2gram:偽情、char_2gram:情報
-> char_3gram:偽情報
-> noun_phrase:偽情報 は追加しない
```

単独名詞へtokenとnoun phraseの両featureを追加すると、namespacesが異なるため
完全一致を不必要に二重強調するからである。

## 9. n-gramは出現位置ごとに数える

同じn-gramが同じ名詞句内の複数位置に現れた場合も、出現ごとにweightを加算する。
名詞句内だけ重複除去する特別規則は設けない。

重複文字列は通常の日本語名詞句では稀であり、OCR重複も対象コーパスではほとんど
観測されていない。正当な反復とOCRノイズをcoreが推測して区別すべきではない。
また、後段の`log1p`がTF増加を緩和する。

## 10. fixed weightを使い、query倍率を使わない

文字列長によるboostは使わず、feature種別ごとの固定weightを使う。

```text
token          1.00
synonym        1.00
noun_phrase    1.00
char_2gram     0.05
char_3gram     0.15
```

長い語は既に多くのn-gramを持ち、珍しい語はQdrant IDFが高くなりやすい。
文字列長boostをさらに掛けると長い語を重複して優遇し、長いOCR誤認語まで強くする。

2-gramは偶然一致が多いため弱くし、3-gramはより具体的なので少し強くする。
token、synonym、noun phraseを同じ`1.0`にしても、同一語の完全一致では複数featureが
積み重なるため、synonymだけの一致と同じscoreにはならない。

query vector全体の一律倍率は使わない。pure sparse rankingでは全scoreをほぼ同率に
変えるだけであり、denseとの相対比を変える目的ならapplicationのfusion policyで
明示的に調整すべきだからである。

## 11. `log1p + Qdrant IDF`だけを使う

同じfeature名のraw weightを合計し、`log1p`を適用する。

```text
raw weight合計 -> log1p -> hash衝突を集約 -> Qdrant Modifier.IDF
```

BM25風のTF飽和・文書長補正は使わない。BM25風と呼ばれる式には複数の選択肢があり、
文書長をtoken数、文字数、異なるfeature数のどれとするかでも結果が変わる。
参照実装の「異なるsparse index数」はhash衝突やfeature規則にも左右され、自然な
文書長とは限らない。必要性が検索評価で確認された場合だけ、別algorithmとして
設計する。

## 12. synonym group IDを独立featureとして使う

元のtoken featureを必ず残し、Sudachiが返したgroup IDを一つずつ独立して追加する。

```text
研究所 -> token:研究所、synonym:860、synonym:6336
ラボ   -> token:ラボ、synonym:859、synonym:860
```

複数IDを一つの組としてhashしない。上の例では`研究所`と`ラボ`が
`synonym:860`で一致する。

synonymは名詞だけに限定せず、POS filterを通過した全morphemeについて、IDがあれば
追加する。固定辞書では形容詞や接続詞にもIDがあり、保持すると決めた内容語の
言い換えを品詞によらず一貫して利用するためである。辞書由来の誤一致はoffline評価で
確認する。

## 13. POS filterは最小限にし、語彙stopwordを使わない

token featureから除外する品詞は次だけとする。

```text
空白
記号
補助記号
助詞
感動詞
```

空白、記号、補助記号はfeatureを作らず、名詞句境界として働く。助詞は高頻度で、
bag-of-featuresでは文法関係を十分に表せないため除外する。感動詞は講演transcriptの
相づちや挨拶として多く、内容検索への寄与が小さいため除外する。

次の品詞は保持する。

```text
助動詞、接続詞、連体詞、代名詞、動詞、形容詞、形状詞、副詞、
名詞、接頭辞、接尾辞、その他
```

助動詞を保持するのは、`利用できる`と`利用できない`の違いに必要な`ない`などを
失わないためである。接続詞は規程文書の`または`、`および`、`ただし`などで意味を
持つ。連体詞を一括除外すると`大きな`、`小さな`、`あらゆる`なども失う。

`これ・この・こんな・こういう`などの指示表現を列挙する語彙stopwordは使わない。
複数morphemeの既知パターンまで網羅するとcoreが複雑になり、解析変化への追従も
必要になる。頻出する不要語の抑制はQdrant IDFへ任せ、実害が確認された場合だけ
再検討する。

## 14. hashはBlake2b-32を使う

complete namespaced feature stringをUTF-8でencodeし、Blake2bの4-byte digestを
unsigned integerとして使う。

```text
hash: blake2b
digest_size: 4 bytes
byte order: big endian
index range: 0 ... 2^32 - 1
```

Qdrantのsparse indexは`uint32`であり、sparse vectorは出現したindexとvalueだけを
保持する。`2^20`へ狭めてもdense配列のような保存量削減にはならず、feature語彙が
10万以上になると衝突が大きく増える。したがって32-bit範囲をすべて使う。

異なるfeature名が同じindexへ衝突した場合は、`log1p`後のvalueを加算する。
最終vectorはindex昇順にsortし、再現可能な出力にする。

## 15. Sudachiと辞書versionを厳密に固定する

```text
sudachipy==0.6.10
sudachidict-core==20260116
split_mode: C
```

辞書versionが変わると、分割、品詞、正規形、synonym IDが変わり得る。コードが同じでも
vector互換性を失うため、version rangeは許可しない。Sudachiまたは辞書を更新する場合は、
新しいalgorithm ID、golden vector、collection、全件再インデックスが必要になる。

## 16. tokenizerは共有し、tokenize呼び出しをlockする

encoderごとに一つのtokenizerを生成し、Sudachiのtokenize呼び出しだけをlockで
保護する。tokenizerと辞書の初期化を一度に抑え、安全性とmemory使用量を優先する。
feature生成とhash処理はlock外で実行できる。

thread-local tokenizerは、実運用でtokenizationがbottleneckだと確認され、benchmarkで
改善が示された場合だけ再検討する。

## 17. `encode()`と`explain()`を分離する

通常APIはvectorだけを返す。

```python
vector = encoder.encode(text)
```

診断時だけ明示的なAPIを使う。

```python
explanation = encoder.explain(text)
```

`explain()`は少なくとも表層形、正規形、品詞、synonym ID、feature種別、feature名、
source morphemeまたは名詞句、raw weight、hash index、衝突集約後のvectorを返す。
通常のQdrant登録・検索経路へ不要な診断情報を持ち込まないためである。

## 18. 位置・語順・否定の係り先はrerankingへ任せる

`sudachi-sparse-v1`は位置を持たないbag-of-featuresであり、高recallの候補取得を責務とする。

```text
query: 利用できない
features: token:利用、token:出来る、token:ない
```

文書`利用できるが、持って帰ることはできない`にも同じtokenが存在するため、sparse
だけでは`ない`の係り先を判断できない。

token sequenceや述語句featureはv1へ追加しない。sequenceを加えても一部のshingleが
別位置で一致し得るうえ、feature数、posting数、weight調整を増やし、否定スコープを
完全には解決しない。特定の述語文法も、`利用可能`と`利用できる`のような意味的な
言い換えを扱えない。

```text
dense + sparseで候補取得
-> score fusion
-> 元テキストを使ったapplication側reranking
-> 最終結果
```

tokenの順序・距離、同じ文や節にあるか、否定、係り受け、意味的関連性はrerankerの
責務とする。sparse encoder自体には組み込まない。パッケージが別途提供するローカル字句
rerankerは、取得済み候補の元chunk文字列だけを対象とし、sparse algorithm IDへ影響しない。

## 19. 処理契約

1. applicationが文書構造とOCR補正を行い、fieldごとのtextとweightを渡す。
2. fieldをまたいでmorpheme、名詞句、n-gramを接続しない。
3. 各fieldをSudachi `SplitMode.C`でtokenizeし、表層形・正規形・品詞・synonym IDを保持する。
4. POS filterを適用し、通過したmorphemeから正規形tokenとsynonym featureを作る。
5. pregroupingなしで厳密な名詞句文法を適用する。
6. 名詞句表層形をNFKC＋case foldingし、noun phraseと2/3-gram featureを作る。
7. feature出現ごとの固定weightへfield weightを掛け、同じfeature名ごとに合計する。
8. 合計weightへ`log1p`を適用する。
9. complete namespaced feature stringをBlake2b-32でhashする。
10. hash衝突したvalueを加算し、index昇順で`SparseVector`を返す。
11. Qdrant collection側が`Modifier.IDF`を適用する。

documentとqueryには同じ1〜10の規則を使う。queryだけの倍率や特別なfeatureは持たない。

## 20. algorithm IDとversioning

短いalgorithm IDは次とする。

```text
sudachi-sparse-v1
```

collection名など、運用上短い識別子が必要な場所で使用する。詳細な互換性情報は別に
公開する。

```text
algorithm_id: sudachi-sparse-v1
sudachipy: 0.6.10
dictionary: sudachidict-core 20260116
split_mode: C
hash: blake2b-32
```

次のいずれかを変えた場合は同じalgorithm IDを使い続けず、全件を新collectionへ
再インデックスする。

- Sudachiまたは辞書version
- split mode
- tokenまたは名詞句の文字正規化
- POS filterまたは語彙stopword
- 名詞句文法、pregrouping、境界規則
- n-gramの長さ、範囲、TF規則
- namespaceまたはsynonym表現
- hash関数、digest size、byte order
- feature weight、query倍率、TF変換
- 文書長補正
- token sequenceなど新しいfeature種別

既存実装はまだ公開・移行・DB作成されていないため、今回確定した仕様を正式な`v1`として
定義し直し、ローカル実装とgolden vectorをこの文書へ合わせて更新した。

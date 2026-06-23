# MotionMillion リポジトリ 全体の作業フロー（初学者向け）

このドキュメントは、`MotionMillion-Codes` リポジトリが「何を・どういう順番で・どう処理しているのか」を、
初めてこのコードを触る人向けにまとめたものです。

対象論文: **"Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data"** (ICCV 2025 Highlight)

---

## 0. このプロジェクトは一言で言うと？

> **テキスト（文章）から人間のモーション（3D の動き）を生成する**システムです。

例：「公園で杖をつきながらゆっくり歩く老人」という文章を入力すると、
それに対応する人体の 3D アニメーション（各関節の動き）を出力します。

技術的には、**大規模言語モデル（LLM）の作り方をモーション生成に応用**しています。
文章生成 LLM が「単語の列」を予測するのと同じ仕組みで、本システムは「モーションの“トークン”の列」を予測します。

そのために、本システムは大きく **2 つのモデル** を組み合わせています。

| # | モデル | 役割 | LLM とのアナロジー |
|---|--------|------|--------------------|
| 1 | **FSQ Tokenizer**（モーション・トークナイザ） | 連続的なモーション ⇄ 離散的なトークン（整数 ID）を相互変換する | テキストの「トークナイザ」に相当 |
| 2 | **LLAMA（自己回帰トランスフォーマ）** | テキストを条件として、モーション・トークンの列を 1 個ずつ予測する | 「GPT 本体」に相当 |

この 2 段構成（**離散トークン化 → 自己回帰生成**）が、本リポジトリ全体を貫く中心的な考え方です。

---

## 1. 全体の作業フロー

学習から推論までの流れを 1 枚の図にすると次のようになります。

```
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 0 : データ準備（リポジトリ外。論文の Data Construction Pipeline）       │
│   Web 動画 → 人物検出/追跡 → SMPL 推定 → フィルタリング                       │
│           → 272 次元モーションベクトル + テキスト記述                          │
│   （成果物が HuggingFace で配布される dataset/MotionMillion）                  │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 1 : Tokenizer (FSQ) の学習                                             │
│   train_tokenizer.py                                                        │
│   モーション(272次元) を 再構成 しながら、離散トークンへの圧縮方法を学習       │
│   出力: fsq_net_6000000.pth                                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 2 : 全モーションをトークン列に変換（コード抽出）                          │
│   train_t2m_get_codes.py                                                    │
│   学習済み FSQ を使い、データセット中の全モーションを整数 ID の列に変換し保存   │
│   出力: VQVAE_codebook_65536_FSQ_all/*.npy , all_data.pkl                   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 3 : Text-to-Motion トランスフォーマ (LLAMA) の学習                      │
│   train_t2m_llama.py                                                        │
│   「テキスト → 次のモーション・トークン」を予測するよう学習（次トークン予測）   │
│   出力: motionmillion_7B_all.pth など                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 4 : 推論（テキスト → モーション生成）                                    │
│   inference_single.py / inference_batch.py                                  │
│   テキスト → (任意でLLAMA3.1で言い換え) → T5でエンコード                       │
│        → LLAMAでトークン列を生成 → FSQデコーダでモーションに復元 → 可視化      │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 5 (任意) : 後処理 (足の滑り除去・平滑化)                                 │
│   postprocess/remove_sliding                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 各 STEP と実行スクリプトの対応表

| STEP | やること | エントリポイント | 実行スクリプト |
|------|---------|------------------|----------------|
| 1 | Tokenizer 学習 | `train_tokenizer.py` | `scripts/train/train_tokenizer.sh` |
| 1' | Tokenizer 評価 | `eval_tokenizer.py` | `scripts/eval/eval_tokenizer.sh` |
| 2 | コード抽出 | `train_t2m_get_codes.py` | `scripts/train/train_t2m_get_codes.sh` |
| 3 | T2M モデル学習 | `train_t2m_llama.py` | `scripts/train/train_t2m_3B.sh`, `train_t2m_7B.sh` |
| 3' | T2M モデル評価 | `eval_t2m_llama.py` | `scripts/eval/eval_t2m_3B.sh`, `eval_t2m_7B.sh` |
| 4 | 推論(1件ずつ対話) | `inference_single.py` | `scripts/inference/single_inference/test_t2m_*.sh` |
| 4 | 推論(ベンチマーク一括) | `inference_batch.py` | `scripts/inference/batch_inference/test_t2m_*.sh` |
| 5 | 後処理 | `postprocess/remove_sliding` | `scripts/run_remove_sliding.sh` |

> **重要なポイント**: STEP 1 と STEP 3 は別々に学習します。
> まず Tokenizer（STEP 1）を完成させ、それを「固定」した状態で
> STEP 2 でデータをトークン化し、STEP 3 で生成モデルを学習します。
> STEP 3 の学習中は Tokenizer の重みは更新されません（`net.eval()` で固定）。

---

## 2. データセットの構造

### 2.1 ディレクトリ構造

データセットは `dataset/MotionMillion/` 以下に配置します（README より）。

```
dataset/
└── MotionMillion/
    ├── motion_data/
    │   └── vector_272/          ← モーション本体。1ファイル=1モーション(.npy)
    │       ├── xxxxx.npy        ← shape = (フレーム数, 272)
    │       └── ...
    ├── texts/                   ← 各モーションのテキスト記述(.txt)
    │   ├── xxxxx.txt            ← 1行=1キャプション（複数行＝言い換え20種など）
    │   └── ...
    ├── mean_std/
    │   └── vector_272/
    │       ├── mean.npy         ← 正規化用の平均 (272,)
    │       └── std.npy          ← 正規化用の標準偏差 (272,)
    └── split/
        └── version1/
            ├── t2m_60_300/      ← Text-to-Motion 用の分割（60〜300フレーム）
            │   ├── train.txt    ← 学習に使うファイル名一覧
            │   ├── test.txt
            │   ├── val.txt
            │   └── all.txt      ← train+val+test 全部（7B-all はこれで学習）
            └── tokenizer_96/    ← Tokenizer 学習用の分割（96フレーム窓）
                ├── train.txt
                ├── test.txt
                └── val.txt
```

ポイント：

- `motion_data` と `texts` は**ファイル名（ID）で対応**しています（例：`abc.npy` ⇔ `abc.txt`）。
- `split/*.txt` には「そのデータ分割に含めるファイル名（拡張子なし）」が1行ずつ書かれています。
- Tokenizer 用 (`tokenizer_96`) と T2M 用 (`t2m_60_300`) で**分割が別々**なのは、
  - Tokenizer は「短い窓（96 フレーム）」でモーションの再構成を学習し、
  - T2M は「文章 1 つに対応する 60〜300 フレームの 1 シーケンス全体」を扱うためです。

### 2.2 モーション 1 フレームの中身（272 次元ベクトル）

このプロジェクトの核となるデータ形式が **`vector_272`** です。
モーションは「フレームごとの 272 次元ベクトルの列」として表現されます（=`(T, 272)` の行列）。

論文 4.1 節「Motion Representations」によると、各フレーム $x_i$ は次の要素を連結したものです
（$N=22$ 関節）。実装上の並び順は `inference_single.py:80-140` の復元コードから読み取れます。

| 次元の範囲 | 記号 | 内容 | サイズ |
|------------|------|------|--------|
| `[0:2]` | $\dot r_x, \dot r_z$ | ルート（腰）の XZ 平面上の並進速度 | 2 |
| `[2:8]` | $\dot r_a$ | ルートの向き（ヘディング）の変化量（6D 回転表現） | 6 |
| `[8:74]` | $p_i$ | 各関節のローカル位置（$3 \times 22$） | 66 |
| `[74:140]` | $v_i$ | 各関節のローカル速度（$3 \times 22$） | 66 |
| `[140:272]` | $r_i$ | 各関節のローカル回転（6D 表現、$6 \times 22$） | 132 |
| | | **合計** | **272** |

なぜこの表現を使うのか（論文より、初学者向けに噛み砕くと）：

- **位置と回転の両方**を持つので、Inverse Kinematics（逆運動学）なしで SMPL/BVH に変換できる。
- 位置と回転が同じ骨格構造から来ているため、互いに整合性のチェック（mutual regularization）が効く。
- 速度などの冗長な情報を持たせることで、学習が安定する。
- HumanML3D 形式にあった回転表現の誤りを修正してある。

> なお `motion_type=vector_272` は実行スクリプトの `--motion_type` 引数で指定されます。
> KIT データセットなら 251 次元など、データセットによって次元数は変わります（`models/vqvae.py:37` 参照）。

### 2.3 正規化（Z-normalization）

モデルに入れる前に、全モーションは平均 0・分散 1 に正規化されます。

```python
motion = (motion - mean) / std        # 正規化（学習・推論の入力時）
motion = motion * std + mean          # 逆正規化（出力を元のスケールに戻す inv_transform）
```

`mean.npy` / `std.npy` は `dataset/MotionMillion/mean_std/vector_272/` にあります
（`dataset/dataset_VQ.py:48-49`、`inference_single.py:356-359`）。

---

## 3. モデル①：FSQ Tokenizer（モーション・トークナイザ）

### 3.1 役割

連続的なモーション（272 次元ベクトルの列）を、**離散的な整数 ID の列**（トークン）に変換し、
また逆に整数 ID の列からモーションを**復元**するモデルです。
画像生成の VQ-VAE や、テキストの BPE トークナイザに相当します。

このトークン化があるおかげで、後段の LLAMA は「整数の列の予測問題」として
モーション生成を解けるようになります。

### 3.2 構成（VQ-VAE 型のオートエンコーダ）

実装は `models/vqvae.py` の `HumanVQVAE` → `VQVAE_251` クラスです。
中身は **Encoder → Quantizer(FSQ) → Decoder** の 3 段です。

```
                ┌──────── FSQ Tokenizer (HumanVQVAE) ────────┐
                │                                            │
 モーション      │  ┌─────────┐  ┌─────────┐   ┌─────────┐    │   再構成された
 (T, 272)  ───► │  │ Encoder │─►│   FSQ    │─►│ Decoder │   │──►  モーション
                │  │ (1D CNN)│  │Quantizer │  │ (1D CNN)│    │   (T, 272)
                │  └─────────┘  └─────────┘   └─────────┘    │
                │       ▲             │             ▲        │
                │   wavelet変換   整数ID列       inverse      │
                │  (Patcher1D)   (トークン)    wavelet変換    │
                └────────────────────────────────────────────┘
```

#### (a) Encoder / Decoder（`models/encdec.py`）

- **1 次元の畳み込み（Conv1d）+ ResNet ブロック**で構成された CNN です。
- 時間方向にダウンサンプリング（`down_t`）して系列を圧縮します。
- 本設定では `down_t=1`（=時間を 1/2 に圧縮）。
- **Wavelet 変換（Haar）**: Encoder の入口で `Patcher1D`、Decoder の出口で `UnPatcher1D` を挟みます
  （`--use_patcher --patch_method haar`）。
  これは論文の重要な工夫で、離散化による**高周波情報の損失（ジッタ＝小刻みな震え）を抑える**ためです。

#### (b) FSQ Quantizer（`models/FSQ.py`）

- **FSQ = Finite Scalar Quantization**（有限スカラー量子化）。
- 通常の VQ-VAE と違い、**コードブック（埋め込みテーブル）を持ちません**。
- 各次元を決められた段階数 `levels` に丸めるだけで離散化します。
  本設定 `nb_code=65536` のとき `levels=[8,8,8,5,5,5]`（$8\times8\times8\times5\times5\times5 = 65536$）。
- コードブックの衝突や崩壊（codebook collapse）が起きにくく、大規模データで安定するのが利点
  （`models/vqvae.py:55-72`）。
- 量子化の勾配は **straight-through 推定**（`round_ste`, `models/FSQ.py:46-49`）で流します。

### 3.3 データの流れ（学習時 / 推論時）

**学習時**（`train_tokenizer.py`）— 自分自身を再構成する「自己教師あり学習」：

```
gt_motion (bs,96,272)
   → net(gt_motion)
   → pred_motion (再構成), perplexity, ...
   → loss = 再構成損失(L1 smooth) + 速度損失(loss_vel) [+ 加速度損失など任意]
   → 逆伝播で Encoder/Decoder を更新（FSQ にはコードブック損失なし）
```

- 損失は `utils/losses.py` の `ReConsLoss`。位置の再構成に加え、速度 `forward_vel` を合わせることで滑らかさを担保します。
- 評価指標は **MPJPE**（関節位置の平均誤差）と **acceleration**（ジッタ量）。
  `eval_tokenizer.py` / `scripts/eval/eval_tokenizer.sh`。

**トークン化したいとき**（`encode`）：

```python
code_idx = net.encode(motion)   # (N, T) の整数 ID。models/vqvae.py:86-103
```

**トークンからモーションに戻すとき**（`forward_decoder`）：

```python
motion = net.forward_decoder(index_motion)   # models/vqvae.py:127-134
```

---

## 4. モデル②：LLAMA 自己回帰トランスフォーマ（Text-to-Motion 本体）

### 4.1 役割

テキスト（の埋め込み）を条件として、**モーション・トークンを 1 個ずつ自己回帰的に予測**するモデルです。
GPT が次の単語を予測するのと同様に、「次のモーション・トークン」を予測します。

実装は `models/lit_llama/model_hf.py` の `LLaMAHF` クラス（LLaMA アーキテクチャ）。

### 4.2 入力の前処理：テキストのエンコード

- テキストは **T5-XL（flan-t5-xl）** エンコーダで**単語レベル**の埋め込み列に変換されます
  （`train_t2m_llama.py:172-180`）。出力次元 `clip_dim=2048`。
- 推論時には、任意で **LLAMA 3.1-8B による「言い換え（rewrite）」**を前段に挟めます
  （`inference_single.py:175-198`）。動画/画像生成と同じく、プロンプトを言い換えて頑健にする手法です。

### 4.3 構成（Hybrid Attention Block の積み重ね）

LLaMA 系の標準的な構成要素を使います（`models/lit_llama/model_hf.py`）。

| 構成要素 | 実装 | 説明 |
|----------|------|------|
| トークン埋め込み | `transformer.wte` (`Embedding`) | モーション・トークン ID → ベクトル |
| テキスト射影 | `llama_proj` (`Linear`) | T5 の埋め込み(2048) → モデル次元へ |
| Block ×N | `Block` | 1ブロック = RMSNorm → 注意機構 → RMSNorm → MLP の残差接続 |
| 注意機構 | `LengthCausalSelfAttention` | **Hybrid/Mixed Attention**（下記） |
| 位置エンコード | `build_rope_cache` / `apply_rope` | **RoPE**（回転位置埋め込み） |
| 正規化 | `RMSNorm` | LayerNorm の代わり。学習を安定化 |
| MLP | `MLP` | SwiGLU（`silu(c_fc1(x)) * c_fc2(x)`） |
| 出力ヘッド | `lm_head` (`Linear`) | 各位置で次トークンの確率（語彙 = `nb_code+2`） |

**Hybrid (Mixed) Attention** が論文の工夫です（`model_hf.py:553-590`、論文 4.3 節）：

- **テキスト部分の単語同士 → 双方向（bidirectional）** に注意できる。
- **モーション・トークン同士 → 因果的（causal、未来は見られない）**。
- これを 1 つの attention mask で表現しています（下三角マスク `OR` テキスト全結合マスク）。

```
語彙サイズ = nb_code + 2 = 65538
   ├─ 0 .. 65535 : 通常のモーション・トークン (FSQ の 65536 種)
   ├─ 65536      : 系列終端トークン (End)
   └─ 65537      : パディング (Pad)
```
（`train_t2m_llama.py:216`、`dataset/dataset_TM_train_motionmillion.py:55-56`）

### 4.4 モデルサイズ（3B / 7B）

`LLaMAHFConfig.from_name()` でサイズを切り替えます（`model_hf.py:31-44`）。

| 名前 | レイヤ数 | ヘッド数 | 埋め込み次元 |
|------|---------|---------|--------------|
| `3B` | 24 | 32 | 3200 |
| `7B` | 36 | 32 | 4096 |

論文では 1B〜7B までスケールさせ、**大きいほど zero-shot 性能（未知の文章への汎化）が上がる**ことを示しています。

### 4.5 データの流れ（学習時 / 推論時）

**学習時**（`train_t2m_llama.py`、次トークン予測）：

```
バッチ = (caption, m_tokens, ..., feat_clip_text, y_mask, ...)
   feat_clip_text : T5 で得たテキスト埋め込み列
   m_tokens       : STEP 2 で作った正解のモーション・トークン列（Pad/End 付き）

   cls_pred = trans_encoder(m_tokens, feat_clip_text, y_mask)   # 各位置の予測
   # 1つずらして「次トークン」を当てる
   loss = CrossEntropy(cls_pred[:-1], m_tokens[1:])             # ignore_index=Pad
```

- 損失は **クロスエントロピー**（`train_t2m_llama.py:296, 339`）。Pad トークンは無視。
- テキスト埋め込みは、対応位置のトークン埋め込みを `torch.where` で**置き換える**形で系列の先頭に差し込まれます（`model_hf.py:159-183`）。

**推論時**（`LLaMAHF.sample`, `model_hf.py:102-136`）：

```
空の列から開始
 → forward_sample でロジット計算 → 最後の位置の確率から次トークンを選ぶ(topk or sample)
 → 列に追加 → 繰り返し（End トークンが出たら、または最大長で停止）
 → 得られたトークン列 index_motion を返す
```

---

## 5. 2 つのモデルをつなぐ「コード抽出」（STEP 2）

`train_t2m_get_codes.py` は、STEP 1 と STEP 3 の橋渡しをします。

1. 学習済み FSQ (`fsq_net_6000000.pth`) を読み込み、`eval()` で固定。
2. データセットの全モーションを `net.encode()` でトークン列に変換し、
   `dataset/MotionMillion/VQVAE_codebook_65536_FSQ_all/<name>.npy` に保存。
3. 各トークン列の末尾に **End トークン**（`nb_code`）を付与しつつ、
   トークンの出現頻度分布（`*_prob.npy`）も計算・保存。
4. 最後に、テキストとコードをまとめた **`all_data.pkl`** を作成
   （`merge_into_pickle`, `train_t2m_get_codes.py:17-42`）。

STEP 3 のデータローダ（`dataset/dataset_TM_train_motionmillion.py`）は、
この `all_data.pkl` を読み込んで「(テキスト埋め込み, モーション・トークン列)」のペアを供給します。
つまり STEP 3 の学習では、**毎回 FSQ を通す必要がなく**、事前計算したトークンを使うので高速です。

---

## 6. 推論パイプライン全体（STEP 4）の詳細

`inference_single.py` を例に、テキスト 1 件を入れてから動画が出るまで：

```
1. テキスト入力 (例: "A man is walking forward")
2. [任意] LLAMA3.1-8B で言い換え          ... call_llama_rewrite()
3. T5-XL でテキスト埋め込み feat_clip_text に変換
4. LLAMA で自己回帰生成 index_motion = trans_encoder.sample(...)   ← モーション・トークン列
5. FSQ デコーダで復元 pred_pose = net.forward_decoder(index_motion)  ← 正規化された272次元
6. 逆正規化 inv_transform(pred_pose, mean, std)                     ← 物理スケールへ
7. 272次元 → 関節回転/位置へ復元 recover_from_local_rotation()       ← SMPL-X 85次元など
8. SMPL-X で 3D 関節を計算し、可視化して .mp4/.gif 保存             ← visualize_smplx_85()
```

- ステップ 7 の復元関数（`recover_from_local_position` / `recover_from_local_rotation`,
  `inference_single.py:80-140`）が、2.2 節で説明した 272 次元の並びを逆にたどって
  グローバルな位置・回転を組み立て直しています。
- `inference_batch.py` は、`assets/infer_batch_prompt` にある **MotionMillion-Eval（126 プロンプト）**
  を一括で処理するベンチマーク用です。

---

## 7. まとめ：依存関係を一望する

```
            [STEP1]                 [STEP2]                  [STEP3]
  motion ──► FSQ学習 ──► fsq.pth ──► 全データをトークン化 ──► all_data.pkl ──► LLAMA学習 ──► t2m.pth
  (272次元)  (再構成)               (encode + End/頻度)      (text+tokens)     (次トークン予測)
                                                                                   │
  テキスト ──► T5-XL ─────────────────────────────────────────────────────────────┘ (条件入力)

            [STEP4 推論]
  テキスト ─►(言い換え)─► T5 ─► LLAMA(sample) ─► tokens ─► FSQ.decode ─► 272次元 ─► SMPL-X ─► 動画
                                t2m.pth              fsq.pth
```

| 用語 | 意味（初学者向け） |
|------|---------------------|
| Tokenizer / FSQ | モーション ⇄ 整数 ID を変換する圧縮器 |
| トークン | モーションを表す整数 ID（語彙 65536 + End/Pad） |
| 自己回帰 (autoregressive) | 直前までの出力をもとに次を 1 個ずつ予測する生成方式 |
| wavelet 変換 | 周波数成分に分けて高周波の損失（震え）を抑える前処理 |
| Hybrid Attention | テキストは双方向・モーションは因果的に見る注意機構 |
| zero-shot | 学習で見ていない文章に対しても妥当なモーションを生成できる能力 |
| MPJPE | 関節位置の平均誤差（再構成・生成の品質指標、小さいほど良い） |
| FID | 生成分布と本物分布の距離（小さいほど良い） |

---

### 参考：主要ファイルの場所

| 種類 | パス |
|------|------|
| Tokenizer 本体 | `models/vqvae.py`, `models/encdec.py`, `models/FSQ.py`, `models/modules.py`(wavelet) |
| LLAMA 本体 | `models/lit_llama/model_hf.py` |
| データローダ | `dataset/dataset_VQ.py`(Tokenizer用), `dataset/dataset_TM_train_motionmillion.py`(T2M用), `dataset/dataset_tokenize.py`(コード抽出用) |
| 学習スクリプト | `train_tokenizer.py`, `train_t2m_get_codes.py`, `train_t2m_llama.py` |
| 推論スクリプト | `inference_single.py`, `inference_batch.py` |
| 評価スクリプト | `eval_tokenizer.py`, `eval_t2m_llama.py` |
| 引数定義 | `options/option_vq.py`, `options/option_transformer.py` |
| 後処理 | `postprocess/remove_sliding/` |
```

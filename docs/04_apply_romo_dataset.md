# RoMo データセットをこのリポジトリで学習する方法（初学者向け・実データ確認版）

このドキュメントは、CVPR 2026 の論文 **「RoMo: A Large-Scale, Richly Organized Dataset and
Semantic Taxonomy for Human Motion Generation」** で公開された **RoMo データセット** を、
本リポジトリ `MotionMillion-Codes` に取り込んで学習するために
「**何を・どこで・どう変えなければならないのか**」を、
**HuggingFace 上の実際の配布物（`https://huggingface.co/RoMoDataset`）を確認したうえで**整理したものです。

前提知識として、先に [`01_repository_workflow.md`](./01_repository_workflow.md)
（リポジトリ全体の流れ）を読んでおくことを強く推奨します。

> このドキュメントは 2026-07-05 に HuggingFace の配布形態を確認して改訂しました。
> 各データセットの内容は更新される可能性があるため、実行前に必ず本家ページの README を再確認してください。

---

## 0. 結論を先に（4行まとめ）

1. RoMo は **前処理済みで4形式（`RoMo-SMPL` / `RoMo-HML-263` / `RoMo-272` / `RoMo-SOMA-77`）**を配布している。
   → 論文だけ読むと「SMPL しか無い」と誤解するが、**実際には 272 次元版（`RoMo-272`）が存在する**。
2. ただし **`RoMo-272` の "272" は本リポジトリの "272"（`vector_272`）と中身の並びが違う**（同じ次元数の別物）。
   そのまま `.npy` を置き換えても、**付属の mean/std・事前学習トークナイザ・可視化関数は使えない**。
3. 配布形式は個別 `.npy` ではなく **Parquet シャード（train/val/test）**。
   → **「Parquet → `<id>.npy` + `texts/<id>.txt` + `split/*.txt` へ展開する変換」**が実作業の中心。
4. 現実的な道は2つ：**(A) `RoMo-272` を新しい motion_type として"ゼロから"学習**（速いが可視化に難あり）／
   **(B) `RoMo-SMPL` を本リポジトリの `vector_272` へ変換**（手間はかかるが事前学習・可視化と完全互換）。

---

## 1. そもそも「データを差し替える」とは何をすることか

本リポジトリの学習は、データに関して次の4つの入口しか見ていません
（根拠：`dataset/dataset_VQ.py`, `dataset/dataset_tokenize.py`, `train_t2m_get_codes.py`）。

| 入口 | 実体（`motionmillion` 分岐の場合） | 使うコード |
|------|------|-----------|
| ① モーション本体 | `dataset/MotionMillion/motion_data/vector_272/<id>.npy`（各 `[T, 272]` float32） | `dataset_VQ.py:145`, `dataset_tokenize.py:49` |
| ② テキスト | `dataset/MotionMillion/texts/<id>.txt`（1行＝1キャプション、学習時にランダムで1本選択） | `train_t2m_get_codes.py:32` |
| ③ 分割リスト | `dataset/MotionMillion/split/<version>/<split>.txt`（id を1行ずつ） | `dataset_VQ.py:150` ほか |
| 付随 | `dataset/MotionMillion/mean_std/vector_272/{mean,std}.npy`（各 `[272]`） | `dataset_VQ.py:148`（Z正規化） |

つまり「RoMo で学習する」とは、**RoMo を上記①〜③＋mean/std の形に変換して置き直す**こと、基本的にはそれだけです。
モデルのコードはほとんど触りません（後述のハードコードパス修正を除く）。

---

## 2. RoMo が HuggingFace で配布している"実物"（ここが今回の確認ポイント）

`https://huggingface.co/RoMoDataset` には、**同じモーションを異なる特徴表現に前処理した4つのデータセット**が置かれています。

| データセット | 表現 | 次元 | 形式 | 本リポジトリとの関係 |
|--------------|------|------|------|---------------------|
| **RoMo-SMPL** | SMPL パラメータ（回転＋並進） | — | Parquet | **道(B)の入力**。`vector_272` へ自前変換する材料 |
| **RoMo-HML-263** | HumanML3D 標準表現 | 263 | Parquet | 本リポジトリの `t2m` 分岐（263次元）に流用可能。要ゼロから学習 |
| **RoMo-272** | HumanML3D 派生の拡張表現 | **272** | Parquet | **道(A)の入力**。次元数は一致するが**並びが別物**（下記2.2） |
| **RoMo-SOMA-77** | SOMA 系 77次元表現 | 77 | Parquet | 本リポジトリは非対応（参考） |

共通仕様（`RoMo-272` / `RoMo-HML-263` の README より）：

- **配布形式**：`data/` 以下に `train-*.parquet` / `val-*.parquet` / `test-*.parquet`（zstd 圧縮シャード）。
  個別 `.npy` ではない。
- **正規化統計**：リポジトリ直下に `Mean.npy` / `Std.npy`（**公式 train 全フレームから算出済み**）。
- **モーション列**：Parquet の `motion` カラムに `list<list<float32>>`（形状 `(T, 272)` など）で格納。
- **テキスト**：`caption_l0` 〜 `caption_l4` の **5階層キャプション**（3〜5語のタグ〜150〜300語の段落まで）。
- **タクソノミ**：`category` / `subcategory` / `atomic_action` の3階層ラベル。
- **規模（RoMo-272）**：合計 **813,931** シーケンス（train 691,982 / val 81,271 / test 40,678）、**30 fps**。
- **ライセンス**：**CC BY-NC 4.0**（非商用）。

読み込みは HuggingFace `datasets` ライブラリを想定：

```python
from datasets import load_dataset
import numpy as np
ds = load_dataset("RoMoDataset/RoMo-272")
sample = ds["train"][0]
motion = np.asarray(sample["motion"], dtype=np.float32)   # (T, 272)
```

### 2.1 本リポジトリ側の `vector_272` の正確な中身（実コードで確認）

本リポジトリの 272 次元は、`utils/motion_process.py` の `recover_from_local_position()` /
`recover_from_local_rotation()` から**厳密に**次のように読めます（`njoint = 22`）。

| 区間 (index) | 次元数 | 内容 | 根拠 |
|--------------|--------|------|------|
| `[0:2]`    | 2   | ルートの水平速度（XZ、heading 除去後） | `final_x[:, :2]` (`motion_process.py:29,61`) |
| `[2:8]`    | 6   | グローバル heading の差分回転（6D） | `final_x[:, 2:8]` (`:30,60`) |
| `[8:74]`   | 66  | 各関節のローカル3D位置（22×3、heading 除去後） | `final_x[:, 8:8+3*njoint]` (`:28,62`) |
| `[74:140]` | 66  | 各関節のローカル速度（22×3） | `8+3*njoint : 8+6*njoint` |
| `[140:272]`| 132 | 各関節のローカル回転（22×6、6D） | `final_x[:, 8+6*njoint:8+12*njoint]` (`:59`) |

合計 = 2 + 6 + 66 + 66 + 132 = **272**。**足接地フラグは持たない**。関節は **22**。

### 2.2 ⚠️ 決定的な注意：RoMo-272 の "272" ≠ 本リポジトリの "272"

- **RoMo-272**（README より）：`[0:262]` は **HumanML3D-263 レイアウト**
  （root角速度 / root線速度 / root高さ / 関節位置 / 6D回転 / 足接地）で、
  `[263:271]` に**9次元の絶対/グローバル特徴**を追加した「**263 + 9 = 272**」。
- **本リポジトリ**：上記 2.1 の「**2 + 6 + 66 + 66 + 132**」で、**足接地なし・root高さや角速度を独立に持たない**別構成。

→ **次元数は同じ 272 でも、各次元の意味・順序がまったく違います。** したがって：

- RoMo-272 の配列をそのまま `vector_272/<id>.npy` に置いても、
  **同梱の事前学習トークナイザ `fsq_net_6000000.pth`（MotionMillion-272 で学習）は使えません**（分布も並びも別）。
- **可視化 `recover_from_local_*()` も誤動作**します（MotionMillion-272 の並び前提のため）。
- RoMo-272 で学習する場合は、**トークナイザから全部ゼロ学習**が前提になります（道A）。

---

## 3. 進め方の分岐：道A / 道B / 道C

| 道 | 入力 | 何をする | 長所 | 短所 |
|----|------|----------|------|------|
| **A（推奨・最短で"学習"まで）** | `RoMo-272` | Parquet を展開し、**新しい motion_type として"ゼロから"学習** | 変換が単純（並べ替えのみ）、mean/std 同梱、実装が最少 | 事前学習ckpt流用不可＝**フルスクラッチ学習**。`recover_*` で**可視化できない**（別デコーダが必要） |
| **B（完全互換）** | `RoMo-SMPL` | SMPL → 本リポジトリ `vector_272` へ変換 | `recover_*`・事前学習トークナイザ初期化・評価器と**完全互換**、可視化可 | **SMPL→272 変換スクリプトの自作が必要**（座標系・関節マッピングが難所） |
| **C（別案）** | `RoMo-HML-263` | 本リポジトリの `t2m` 分岐（263次元）で学習 | HumanML3D の成熟した可視化資産が使える | 本リポジトリの事前学習は272系。263は**ゼロ学習**。リポジトリ主眼から外れる |

> **推奨方針**：まず**道A**でパイプライン全体（展開→トークナイザ→get_codes→T2M）を小サブセットで通し、
> 生成が回ることを確認する。可視化・事前学習流用・厳密評価まで必要になった段階で**道B**を検討する。

以降は**道A**を主軸に手順を示し、道B/Cは差分だけ補足します。

---

## 4. 目標ディレクトリ構成（道A）

RoMo-272 は本家 272 と別物なので、**別の motion_type 名**（例：`vector_272_romo`）で置くと事故を防げます。
最も手数が少ないのは `dataset/MotionMillion/` を間借りして `motionmillion` 分岐を再利用する方法です。

```
dataset/
└── MotionMillion/                       # 分岐 'motionmillion' を再利用（別名にするなら §7 のコード追加が必要）
    ├── motion_data/
    │   └── vector_272_romo/             # ★RoMo-272 専用の motion_type（本家 vector_272 と混ぜない）
    │       ├── 000000.npy               # 各 [T, 272] float32（RoMo-272 の並び）
    │       └── ...
    ├── texts/                           # ※ 道A/B/C 共通で texts_romo にすると安全
    │   ├── 000000.txt                   # 5行（caption_l0〜l4）
    │   └── ...
    ├── mean_std/
    │   └── vector_272_romo/
    │       ├── mean.npy                 # RoMo 同梱 Mean.npy をコピー/リネーム [272]
    │       └── std.npy                  # 同 Std.npy [272]
    └── split/
        └── romo_v1/
            ├── tokenizer_96/            # トークナイザ学習用（≥96 フレーム）
            │   ├── train.txt
            │   ├── val.txt
            │   └── test.txt
            └── t2m_60_300/              # T2M 学習用（60〜300 フレーム目安）
                ├── train.txt
                ├── val.txt
                ├── test.txt
                └── all.txt              # get_codes が参照（§6 STEP D）
```

---

## 5. 事前準備（依存関係とダウンロード）

### 5.1 依存ライブラリ

RoMo は Parquet 配布なので、`datasets` / `pyarrow` / `huggingface_hub` が必要です
（本環境のシステム python には未導入。学習用の conda 環境に入れてください）。

```bash
# 学習用の環境を有効化してから
pip install -U "datasets>=2.19" pyarrow huggingface_hub
```

### 5.2 データ取得

CC BY-NC 4.0 のためログイン/規約同意が必要な場合があります。

```bash
huggingface-cli login          # 必要なら
# 方法1: リポジトリごと取得（Parquet と Mean/Std を丸ごと）
huggingface-cli download RoMoDataset/RoMo-272 --repo-type dataset \
  --local-dir dataset/_download/RoMo-272
# 方法2: 5.3 のスクリプトで load_dataset を使う場合はキャッシュに自動取得
```

> ⚠️ **容量**：81万シーケンス・約1,238時間分。Parquet でも**数十GB規模**になり得ます。
> 現在ルート空きは約224GB。まず `--split train[:1%]` 等の小サブセットで検証してから全量を取得してください。

### 5.3 Parquet → 本リポジトリ形式へ展開するスクリプト（★リポジトリに無い＝要自作）

以下は**道A用の最小実装例**です（`scripts/prepare/export_romo272.py` などとして保存）。
`.npy` 本体・`texts/<id>.txt`（5行）・`split/*.txt`・`mean/std` を一括生成します。

```python
import os, numpy as np
from os.path import join as pjoin
from datasets import load_dataset

OUT   = "dataset/MotionMillion"
MTYPE = "vector_272_romo"
TXT   = "texts_romo"
VER   = "romo_v1"
os.makedirs(pjoin(OUT, "motion_data", MTYPE), exist_ok=True)
os.makedirs(pjoin(OUT, TXT), exist_ok=True)
for sub in ("tokenizer_96", "t2m_60_300"):
    os.makedirs(pjoin(OUT, "split", VER, sub), exist_ok=True)
os.makedirs(pjoin(OUT, "mean_std", MTYPE), exist_ok=True)

ds = load_dataset("RoMoDataset/RoMo-272")     # 検証時は split="train[:1%]" 等に
splits = {"train": "train", "validation": "val", "test": "test"}  # HF側キー→本リポジトリ表記

tok_ids = {"train": [], "val": [], "test": []}
t2m_ids = {"train": [], "val": [], "test": []}

for hf_split, name in splits.items():
    if hf_split not in ds:               # val が "validation" でない場合に備える
        hf_split = "val" if name == "val" else hf_split
    for i, s in enumerate(ds[hf_split]):
        mid = f"{name}_{i:06d}"          # 一意な id 命名（衝突しなければ何でもよい）
        motion = np.asarray(s["motion"], dtype=np.float32)   # (T, 272)
        np.save(pjoin(OUT, "motion_data", MTYPE, mid + ".npy"), motion)

        caps = [s.get(f"caption_l{k}", "") for k in range(5)]
        caps = [c.replace("\n", " ").strip() for c in caps if c and c.strip()]
        with open(pjoin(OUT, TXT, mid + ".txt"), "w") as f:
            f.write("\n".join(caps) + "\n")

        T = motion.shape[0]
        if T >= 96:                      # tokenizer は window=96 未満を弾く
            tok_ids[name].append(mid)
        if 60 <= T <= 300:               # T2M の想定レンジ（必要に応じ調整）
            t2m_ids[name].append(mid)

def dump(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")

for name in ("train", "val", "test"):
    dump(pjoin(OUT, "split", VER, "tokenizer_96", name + ".txt"), tok_ids[name])
    dump(pjoin(OUT, "split", VER, "t2m_60_300",  name + ".txt"), t2m_ids[name])
# get_codes が参照する all.txt（t2m 全 id）
dump(pjoin(OUT, "split", VER, "t2m_60_300", "all.txt"),
     t2m_ids["train"] + t2m_ids["val"] + t2m_ids["test"])

# 正規化統計：RoMo 同梱 Mean/Std をコピー（无ければ自前算出）
import shutil
src = "dataset/_download/RoMo-272"
if os.path.exists(pjoin(src, "Mean.npy")):
    shutil.copy(pjoin(src, "Mean.npy"), pjoin(OUT, "mean_std", MTYPE, "mean.npy"))
    shutil.copy(pjoin(src, "Std.npy"),  pjoin(OUT, "mean_std", MTYPE, "std.npy"))
print("done")
```

> **カラム名は要確認**：`motion` / `caption_l0..l4` は README 記載の想定名です。実データで
> `print(ds["train"].column_names)` を実行し、`length` や `id`・`category` 等の実際の列名に合わせてください。

---

## 6. 学習パイプライン（道A・コマンド）

パイプラインは3段：**① トークナイザ学習 → ② コード化＋`all_data.pkl` → ③ T2M（LLaMA）学習**。
既存スクリプトの引数を RoMo 用に差し替えて実行します。

### STEP A: トークナイザを"ゼロから"学習

`scripts/train/train_tokenizer.sh` をコピーして RoMo 用に修正（**`--resume-pth` は付けない＝スクラッチ**）。

```bash
# 変更する主な引数（train_tokenizer.sh）
--dataname     motionmillion            # 分岐キー（MotionMillion を間借り）
--motion_type  vector_272_romo          # ★RoMo 用ディレクトリ名に一致させる
--text_type    texts_romo               # §5.3 に合わせる
--version      romo_v1/tokenizer_96     # ★split パスに一致させる
--window-size  96
--nb-code      65536
--quantizer    FSQ
--exp-name     train_VQVAE_FSQ_romo272  # 任意
# ※ --resume-pth は書かない（本家 fsq_net は流用不可）
```

> 実行例（4GPU、`scripts/train/train_tokenizer.sh` を編集後）：
> ```bash
> bash scripts/train/train_tokenizer.sh
> ```
> 学習後、チェックポイント（例：`results/output/FSQ_96len/.../net_last.pth`）のパスを控える。

### STEP B: get_codes のハードコードを RoMo 用に直す

`train_t2m_get_codes.py` にはパスが固定されています。以下を RoMo 用に変更します。

| 変更箇所 | 現状 | RoMo 用 |
|----------|------|---------|
| `train_t2m_get_codes.py:32` | `os.path.join(root_dir, "texts", name+".txt")` | `"texts_romo"` に変更 |
| `train_t2m_get_codes.py:33` | `"VQVAE_codebook_65536_FSQ_all"` | `--vq-name` に合わせる（コード保存先） |
| `train_t2m_get_codes.py:152` | `pjoin(root_dir, "split/version1/t2m_60_300/all.txt")` | `"split/romo_v1/t2m_60_300/all.txt"` に変更 |

### STEP C: コード化＋`all_data.pkl` 生成

`scripts/train/train_t2m_get_codes.sh` を編集し、STEP A のトークナイザを指定して実行。

```bash
--resume-pth   results/output/FSQ_96len/.../net_last.pth   # ★STEP A の成果物
--vq-name      VQVAE_codebook_65536_FSQ_romo               # 任意（STEP B と一致）
--dataname     motionmillion
--motion_type  vector_272_romo
--text_type    texts_romo
--version      romo_v1/t2m_60_300
```
```bash
bash scripts/train/train_t2m_get_codes.sh
# → dataset/MotionMillion/all_data.pkl と VQVAE_codebook_.../<id>.npy が生成される
```

### STEP D: T2M（LLaMA）本体を学習

`scripts/train/train_t2m_3B.sh`（または `train_t2m_7B.sh`）を編集。

```bash
--resume-pth   results/output/FSQ_96len/.../net_last.pth   # ★STEP A のトークナイザ
--dataname     motionmillion
--motion_type  vector_272_romo
--text_type    texts_romo
--version      romo_v1/t2m_60_300
--train_split  train
--text_encode  flan-t5-xl               # checkpoints/models--google--flan-t5-xl を使用
```
```bash
bash scripts/train/train_t2m_3B.sh
```

> **テキストエンコーダ**：`flan-t5-xl`（`checkpoints/models--google--flan-t5-xl`、約43GB）は道A/B/C 共通で必要。
> これは既にダウンロード済み。

---

## 7. 別名 `romo` 分岐にしたい場合（任意）

`dataset/MotionMillion/` を間借りせず `dataset/RoMo/` に独立させたい場合、各ローダの分岐に
`elif dataset_name == 'romo':` を追加します。
**変更ファイル**：`dataset/dataset_VQ.py`、`dataset/dataset_tokenize.py`、
`dataset/dataset_TM_train_motionmillion.py`、（評価するなら）`dataset/dataset_TM_eval_motionmillion.py`、
および `train_t2m_get_codes.py:96` の `root_dir` 分岐。
手数は増えるので、**まずは `motionmillion` 間借り＋別 motion_type 名**を推奨します。

---

## 8. 道B（RoMo-SMPL → 本リポジトリ `vector_272`）の要点

可視化・事前学習トークナイザ初期化・評価器まで完全に使いたい場合はこちら。**変換スクリプトの自作が必要**です。

概念フロー：

```
RoMo-SMPL（24関節 axis-angle + global trans, 30fps）
  ① SMPL Forward Kinematics → 各関節の3Dワールド位置（body_models/ の SMPL+H, utils/smplx）
  ② 24→22 関節へ抽出/並べ替え（HumanML3D 関節順）
  ③ 座標系を HumanML3D 規約へ：Y-up・初フレーム接地・前方 +Z
  ④ heading を剥がし、root水平速度/heading差分/位置/速度/6D回転 を算出 → 272 へエンコード
  ▼
dataset/MotionMillion/motion_data/vector_272/<id>.npy（[T,272]、本家と同じ並び）
```

実装のヒント：

- **逆変換が設計図**：`utils/motion_process.py` の `recover_from_local_position()` /
  `recover_from_local_rotation()` が `272 → 位置/回転` を行う。これを逆向きに読めば順変換（→272）を実装できる。
- 6D/行列/軸角の相互変換：`utils/rotation_conversions.py`、`utils/face_z_align_util.py`
  （`rotation_6d_to_matrix`, `matrix_to_axis_angle` 等）。
- FK のスケルトン：`utils/skeleton.py`、`utils/smplx/`、`body_models/` の SMPL+H。
- ⚠️ **③ 座標系が最大の難所**。少数サンプルで「272→`recover_*`→可視化」の往復検証を必ず行う。
  RoMo は *orientation-aligned* / *body-scale standardized* 済みなので幾分楽だが、**Y/Z 軸入替と接地は要検証**。
- 変換後は **本家の mean/std を使わず** RoMo 分布で再算出（道Aの §5.3 と同様）。
- 道Bなら事前学習トークナイザ `checkpoints/pretrained_models/fsq_net_6000000.pth` を **`--resume-pth` 初期値**に使える。

---

## 9. FPS とクリップ長の注意

- **FPS は一致（30）**：`dataset_tokenize.py` の `motionmillion` 分岐は `fps=30`、RoMo-272 も 30fps。リサンプリング不要。
  （※ `dataset_TM_train_motionmillion.py` の `fps=20` はトークン列段の記述で実フレームレートに影響しない。混同注意。）
- **クリップ長**：tokenizer は 96 フレーム固定クロップ、T2M は `max_motion_length=300`（`block-size 301`）。
  RoMo は **30〜600 フレーム**。96未満は tokenizer split から除外（`dataset_VQ.py` が skip）。
  300超を丸ごと学習対象にしたい場合は `max_motion_length` と位置エンコーディング長の見直しが必要。

---

## 10. ★現在足りないもの（このリポジトリに無い＝準備が必要なもの）

| # | 不足物 | 影響する道 | 対応 |
|---|--------|-----------|------|
| M1 | **Parquet → `<id>.npy`+`texts`+`split` 展開スクリプト** | A/B/C 共通 | §5.3 を `scripts/prepare/` に作成 |
| M2 | **`datasets` / `pyarrow` / `huggingface_hub`** 未インストール | A/B/C 共通 | §5.1 で pip 導入 |
| M3 | **RoMo データ本体（未ダウンロード）** | A/B/C 共通 | §5.2。数十GB、まず小サブセットで検証 |
| M4 | **RoMo-272 用の可視化デコーダ** | **道A** | `recover_*` は本家並び前提で使えない。RoMo-SMPL への逆変換 or HML263 デコーダが別途必要 |
| M5 | **SMPL→`vector_272`（本家並び）変換スクリプト** | **道B** | §8。リポジトリ同梱の逆変換を設計図に自作（最難関） |
| M6 | **`get_codes` 等のハードコードパス修正** | A/B | §6 STEP B（`texts_romo` / split パス / vq-name） |
| M7 | **RoMo 用に"ゼロ学習"したトークナイザ** | **道A/C** | 本家 `fsq_net_6000000.pth` は流用不可。§6 STEP A で新規学習 |
| M8 | **RoMo 分布の評価器（任意）** | 厳密評価時 | `models/evaluator_wrapper_motionmillion_rpr272.py` は MotionMillion 学習済み。RoMo で FID/Matching を厳密に測るなら再学習。**生成モデル学習自体には不要** |
| M9 | **別名 `romo` 分岐（任意）** | 独立配置する場合 | §7。間借りなら不要 |

> **要するに**：論文だけ見ると「SMPL→272 変換（M5）が唯一の壁」に見えますが、
> HuggingFace には前処理済みの `RoMo-272` があるため、**道Aを選べば M5 は不要**になります。
> 代わりに **M4（RoMo-272 の可視化）** と **M7（トークナイザのゼロ学習）** が新たな要対応点として立ち上がります。

---

## 11. 作業チェックリスト

**共通**
- [ ] `datasets`/`pyarrow`/`huggingface_hub` を学習環境に導入（M2）
- [ ] 対象データセットの README とカラム名を実データで確認（`ds["train"].column_names`）
- [ ] 小サブセット（例 `train[:1%]`）でパイプラインを一周させる

**道A（RoMo-272・推奨）**
- [ ] §5.3 の展開スクリプトを作成し、`vector_272_romo/<id>.npy` + `texts_romo/<id>.txt`(5行) + split を生成（M1）
- [ ] RoMo 同梱 `Mean.npy`/`Std.npy` を `mean_std/vector_272_romo/{mean,std}.npy` に配置
- [ ] STEP A: トークナイザを**スクラッチ**学習（`--resume-pth` なし、`--motion_type vector_272_romo`）（M7）
- [ ] STEP B/C: get_codes のパス修正＋コード化＋`all_data.pkl` 生成（M6）
- [ ] STEP D: T2M（3B/7B）を学習
- [ ] 生成結果の確認方法を用意（M4：RoMo デコーダ or SMPL 逆変換で可視化）

**道B（RoMo-SMPL→vector_272・完全互換）**
- [ ] SMPL→`vector_272` 変換を実装し、`272→recover_*→可視化`の往復で正しさ検証（M5）
- [ ] 24→22 関節マッピングと座標系（Y-up/接地/前方Z+）を確認
- [ ] RoMo 分布で mean/std を再算出
- [ ] 事前学習トークナイザを `--resume-pth` 初期化として活用可
- [ ] 以降は本家 `version1/...` と同じフローで学習・可視化・評価

---

## 12. 参考（関連ドキュメント・出典）

- [`01_repository_workflow.md`](./01_repository_workflow.md) — リポジトリ全体の処理フロー
- [`02_paper_chapter4_architecture.md`](./02_paper_chapter4_architecture.md) — モデル構造
- RoMo データ本体: `https://huggingface.co/RoMoDataset`
  （`RoMo-SMPL` / `RoMo-HML-263` / `RoMo-272` / `RoMo-SOMA-77`）
- RoMo プロジェクトページ: `https://davidzhang73.github.io/romo-website/`
- RoMo 論文: `paper/Zhang_RoMo_..._CVPR_2026_paper.pdf`
- 本リポジトリ側の根拠コード:
  - 入口/形式: `dataset/dataset_VQ.py`, `dataset/dataset_tokenize.py`, `train_t2m_get_codes.py`
  - 272 の定義: `utils/motion_process.py`（`recover_from_local_position/rotation`）
  - 変換の手掛かり: `utils/face_z_align_util.py`, `utils/rotation_conversions.py`, `utils/skeleton.py`, `utils/smplx/`
  - 学習スクリプト: `scripts/train/{train_tokenizer,train_t2m_get_codes,train_t2m_3B,train_t2m_7B}.sh`

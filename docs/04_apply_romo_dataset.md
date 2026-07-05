# RoMo データセットをこのリポジトリで学習する方法（初学者向け・motion-toolbox 活用版）

このドキュメントは、CVPR 2026 の論文 **「RoMo: A Large-Scale, Richly Organized Dataset and
Semantic Taxonomy for Human Motion Generation」** の **RoMo データセット** を、
本リポジトリ `MotionMillion-Codes` に取り込んで学習するための実務手順です。
**HuggingFace の実配布物（`https://huggingface.co/RoMoDataset`）と、
RoMo 公式ツールキット `motion-toolbox`（`../motion-toolbox`）を実地確認**したうえで整理しています。

前提として、先に [`01_repository_workflow.md`](./01_repository_workflow.md)（リポジトリ全体の流れ）を読むことを推奨します。

> 改訂履歴：2026-07-05 に `../motion-toolbox` の存在を確認し、方針を大幅に更新。
> 以前の版で「SMPL→272 変換スクリプトの自作が最大の壁」「RoMo-272 は本家272と別物」としていた記述は
> **誤りだったので撤回**します（下記 §2・§3 参照）。実際は公式ツールで変換・可視化できます。

---

## 0. 結論を先に（4行まとめ）

1. **`../motion-toolbox` が RoMo 公式の変換・可視化ツールキット**で、これが全ての鍵。
2. **ツールボックスの「272」= 本リポジトリの `vector_272` と完全同一**
   （`[Root_Lin_Vel 2, Root_Ang_Vel 6, Local_Pos 66, Joint_Vel 66, Rot6D 132]`、22関節）。
   → 前提バージョン間の**次元・並びのギャップは無い**。
3. したがって道は2つとも容易：
   **(1) `RoMo-272` を直接落として展開**（変換ゼロ）／
   **(2) `RoMo-SMPL` を `SMPLTo272Converter` で変換**（公式実装、往復検証済み）。
4. 残る実作業は **Parquet → `<id>.npy` + `texts/<id>.txt` + `split/*.txt` への展開**と、
   **ハードコードパスの微修正**、**（推奨）RoMo 分布でのトークナイザ微調整**だけ。

---

## 1. 本リポジトリの学習が要求する4つの入口（変わらない事実）

学習コードがデータについて見ているのは次の4点だけです
（根拠：`dataset/dataset_VQ.py`, `dataset/dataset_tokenize.py`, `train_t2m_get_codes.py`）。

| 入口 | 実体（`motionmillion` 分岐） | 使うコード |
|------|------|-----------|
| ① モーション本体 | `dataset/MotionMillion/motion_data/vector_272/<id>.npy`（各 `[T, 272]` float32） | `dataset_VQ.py:145`, `dataset_tokenize.py:49` |
| ② テキスト | `dataset/MotionMillion/texts/<id>.txt`（1行＝1キャプション、学習時ランダム選択） | `train_t2m_get_codes.py:32` |
| ③ 分割リスト | `dataset/MotionMillion/split/<version>/<split>.txt`（id を1行ずつ） | `dataset_VQ.py:150` ほか |
| 付随 | `dataset/MotionMillion/mean_std/vector_272/{mean,std}.npy`（各 `[272]`） | `dataset_VQ.py:148`（Z正規化） |

「RoMo で学習する」＝ **RoMo をこの4点の形に置き直す**、それだけです。

### 1.1 本リポジトリ `vector_272` の正確な中身（実コード確認済み）

`utils/motion_process.py`（`recover_from_local_position/rotation`, `njoint=22`）より：

| 区間 | 次元 | 内容 |
|------|------|------|
| `[0:2]`    | 2   | ルート水平速度（XZ、heading 除去後） |
| `[2:8]`    | 6   | グローバル heading 差分回転（6D） |
| `[8:74]`   | 66  | 各関節ローカル3D位置（22×3、heading 除去後） |
| `[74:140]` | 66  | 各関節ローカル速度（22×3） |
| `[140:272]`| 132 | 各関節ローカル回転（22×6、6D） |

合計 **272**、足接地フラグ無し、22関節。

---

## 2. RoMo の実配布物と、motion-toolbox の 272

### 2.1 HuggingFace `RoMoDataset` の4形式

| データセット | 表現 | 次元 | 形式 | 使い道 |
|--------------|------|------|------|--------|
| **RoMo-SMPL** | SMPL パラメータ | — | Parquet | 道(2)の入力。`SMPLTo272Converter` にかける |
| RoMo-HML-263 | HumanML3D 標準 | 263 | Parquet | 本リポジトリ `t2m` 分岐用（今回は非推奨） |
| **RoMo-272** | 272 表現 | **272** | Parquet | 道(1)の入力。**本家272と同一レイアウト** |
| RoMo-SOMA-77 | SOMA 系 | 77 | Parquet | 本リポジトリ非対応 |

共通仕様（README より）：`data/{train,val,test}-*.parquet`（zstd）、直下に `Mean.npy`/`Std.npy`、
`motion`（`list<list<float32>>`）、`caption_l0`〜`caption_l4`（5階層）、
`category`/`subcategory`/`atomic_action`、**30fps**、合計 **813,931** シーケンス、**CC BY-NC 4.0**。

### 2.2 ✅ 重要な確定事項：motion-toolbox / RoMo-272 の272 = 本家272

`../motion-toolbox/docs/272_FORMAT_REFERENCE.md` と
`src/motion_toolbox/converters/motion_converters.py`（`SMPLTo272Converter` docstring）に明記：

```
[Root_Lin_Vel (2), Root_Ang_Vel (6), Local_Pos (66), Joint_Vel (66), Rotations_6D (132)] = 272
# [0:2]=root水平速度, [2:8]=heading6D, [8:74]=位置(22×3), [74:140]=速度(22×3), [140:272]=回転(22×6)
```

- これは §1.1 の本リポジトリ `vector_272` と**完全一致**（22関節・heading正規化・Y-up 正準化）。
- 参照実装は `github.com/Li-xingXiao/272-dim-Motion-Representation`＝**MotionMillion 共著者 Lixing Xiao** の実装。
- サンプル `sample_data/format_272/000000.npy` の形状も `(221, 272)` で整合。

→ **次元数だけでなく意味・順序まで同一**。したがって、
RoMo-272 の配列はそのまま `vector_272/<id>.npy` として本リポジトリに投入でき、
**事前学習トークナイザ `fsq_net_6000000.pth`（本家272で学習）も原理的に流用可能**です
（ただし RoMo は分布が違うので、mean/std は RoMo 側を使い、トークナイザは微調整/再学習が望ましい。§6 STEP A）。

> 以前の版の「263+9 で別物」という記述は、HuggingFace README の自動要約に基づく誤りでした。撤回します。

---

## 3. 変換の切り札：`../motion-toolbox`（RoMo 公式）

RoMo プロジェクトの公式ツールキット。前バージョンで「自作が必要」とした処理が**ほぼ実装済み**です。

| ツールボックスの機能 | 実体 | 本タスクでの用途 |
|----------------------|------|------------------|
| **SMPL → 272** | `converters.SMPLTo272Converter` | **RoMo-SMPL → `vector_272/<id>.npy`**（道2の心臓部） |
| **272 → SMPL** | `converters.Format272ToSMPLConverter` | 生成結果を SMPL に戻す |
| **272 → Motion（可視化用）** | `converters.Format272ToMotionConverter` | 生成/学習データの**可視化**（22/24関節） |
| **272 データセット** | `data.Format272Dataset`（Mean/Std 自動読込） | ローダの参考・検証 |
| **統合可視化** | `visualization.Visualizer`（263/272/205/232 自動判別→HTML） | **可視化の弱点を解消** |
| **HumanML3D → 272** | `scripts/convert_humanml3d_to_272.py` | 参考（AMASS SMPL 経由。263直変換は非対応） |

**検証実績**：`docs/272D_VALIDATION_RESULTS.md` で
`HumanML3D(263)→SMPL→272→SMPL` のラウンドトリップが **7/7 合格**、平均速度誤差 0.001 m/frame。

### 3.1 セットアップ

```bash
cd ../motion-toolbox
uv sync                         # or: pip install -e ".[ml]"
# SMPL⇔272 と 24関節メッシュ可視化には SMPL body model が必要
python scripts/setup_smpl_models.py    # → ~/.motion_toolbox/datasets/body_models_clean_pkl
# importable name は motion_toolbox（アンダースコア）
```

> `SMPLTo272Converter` / 24関節出力 / メッシュ可視化は `body_model_path`（SMPL）必須。
> スケルトンのみの可視化なら body model 不要。

---

## 4. 進め方の分岐（どちらも容易）

| 道 | 入力 | 変換 | 長所 | 留意点 |
|----|------|------|------|--------|
| **(1) RoMo-272 直接（推奨・最短）** | `RoMo-272` | **不要**（Parquet 展開のみ） | 変換ゼロ、mean/std 同梱、可視化はツールボックス | RoMo-272 の並びが本家と一致することを1サンプルで最終確認 |
| **(2) RoMo-SMPL 変換** | `RoMo-SMPL` | `SMPLTo272Converter` | 変換を自分で管理、SMPL も手元に残る | SMPL body model の準備が必要 |

**推奨**：まず道(1)。RoMo-272 を落として展開 → 1サンプルを可視化して整合確認 → 学習。
可視化検証やSMPL併用が必要になったら道(2)/ツールボックスを併用。

---

## 5. 目標ディレクトリ構成

RoMo は本家と同一レイアウトなので `dataset/MotionMillion/` を間借りし `motionmillion` 分岐を再利用できます。
事故防止に**別の motion_type / text_type / version 名**を付けます。

```
dataset/MotionMillion/
├── motion_data/vector_272_romo/     # ★RoMo の [T,272]（本家 vector_272 と混ぜない）
│   ├── train_000000.npy ...
├── texts_romo/                      # 5行（caption_l0〜l4）
│   ├── train_000000.txt ...
├── mean_std/vector_272_romo/
│   ├── mean.npy                     # RoMo 同梱 Mean.npy をコピー [272]
│   └── std.npy                      # 同 Std.npy [272]
└── split/romo_v1/
    ├── tokenizer_96/{train,val,test}.txt     # ≥96 フレーム
    └── t2m_60_300/{train,val,test,all}.txt   # 60〜300 フレーム目安（all は get_codes 用）
```

---

## 6. 手順（道(1) RoMo-272 直接）

### STEP 0: 依存とダウンロード

```bash
# 学習環境で
pip install -U "datasets>=2.19" pyarrow huggingface_hub
huggingface-cli login          # CC BY-NC。必要なら
huggingface-cli download RoMoDataset/RoMo-272 --repo-type dataset \
  --local-dir dataset/_download/RoMo-272
```

> ⚠️ 81万シーケンス（数十GB規模）。まず `--split train[:1%]` 等でパイプラインを通してから全量へ。
> ディスク空きは現在約224GB。

### STEP 1: Parquet → 本リポジトリ形式へ展開（★要自作の小スクリプト）

`scripts/prepare/export_romo272.py` などとして保存（道(1)用の最小実装）：

```python
import os, shutil, numpy as np
from os.path import join as pjoin
from datasets import load_dataset

OUT, MTYPE, TXT, VER = "dataset/MotionMillion", "vector_272_romo", "texts_romo", "romo_v1"
for p in [pjoin(OUT,"motion_data",MTYPE), pjoin(OUT,TXT), pjoin(OUT,"mean_std",MTYPE),
          pjoin(OUT,"split",VER,"tokenizer_96"), pjoin(OUT,"split",VER,"t2m_60_300")]:
    os.makedirs(p, exist_ok=True)

ds = load_dataset("RoMoDataset/RoMo-272")            # 検証時は split="train[:1%]"
name_map = {"train":"train", "validation":"val", "test":"test"}
tok = {"train":[],"val":[],"test":[]}; t2m = {"train":[],"val":[],"test":[]}

for hf_split, name in name_map.items():
    if hf_split not in ds:                            # val キー名の揺れに対応
        hf_split = {"val":"validation"}.get(name, hf_split)
        if hf_split not in ds: continue
    for i, s in enumerate(ds[hf_split]):
        mid = f"{name}_{i:06d}"
        m = np.asarray(s["motion"], dtype=np.float32)  # (T, 272)
        np.save(pjoin(OUT,"motion_data",MTYPE,mid+".npy"), m)
        caps = [str(s.get(f"caption_l{k}","")).replace("\n"," ").strip() for k in range(5)]
        caps = [c for c in caps if c]
        open(pjoin(OUT,TXT,mid+".txt"),"w").write("\n".join(caps)+"\n")
        T = m.shape[0]
        if T >= 96:            tok[name].append(mid)
        if 60 <= T <= 300:     t2m[name].append(mid)

def dump(path, ids): open(path,"w").write("\n".join(ids)+"\n")
for name in ("train","val","test"):
    dump(pjoin(OUT,"split",VER,"tokenizer_96",name+".txt"), tok[name])
    dump(pjoin(OUT,"split",VER,"t2m_60_300", name+".txt"), t2m[name])
dump(pjoin(OUT,"split",VER,"t2m_60_300","all.txt"), t2m["train"]+t2m["val"]+t2m["test"])

src = "dataset/_download/RoMo-272"                     # 同梱 mean/std を流用
if os.path.exists(pjoin(src,"Mean.npy")):
    shutil.copy(pjoin(src,"Mean.npy"), pjoin(OUT,"mean_std",MTYPE,"mean.npy"))
    shutil.copy(pjoin(src,"Std.npy"),  pjoin(OUT,"mean_std",MTYPE,"std.npy"))
print("done")
```

> カラム名（`motion`, `caption_l0..l4`）は実データで `ds["train"].column_names` を確認して合わせること。

### STEP 1.5: ★整合確認（1サンプルだけ・必須）

RoMo-272 の並びが本家と一致するかを、**可視化の往復**で必ず確かめます。

```python
# 本リポジトリ側のデコード
import numpy as np, torch
from utils.motion_process import recover_from_local_position
x = np.load("dataset/MotionMillion/motion_data/vector_272_romo/train_000000.npy")
pos = recover_from_local_position(x, 22)     # [T,22,3] 自然な人体に見えればレイアウト一致
print(pos.shape)
```
```python
# もしくは motion-toolbox で HTML 可視化（../motion-toolbox 環境で）
from motion_toolbox.visualization import Visualizer
Visualizer().visualize(x, "check.html")      # 破綻なく立って動けばOK
```

破綻（横倒し・地面貫通・バラバラ）する場合のみ、道(2)や並べ替えを検討。通常は一致するはずです。

### STEP 2: get_codes のハードコードを RoMo 用に修正

`train_t2m_get_codes.py`：

| 箇所 | 現状 | 変更 |
|------|------|------|
| `:32` | `os.path.join(root_dir,"texts",name+".txt")` | `"texts_romo"` |
| `:33` | `"VQVAE_codebook_65536_FSQ_all"` | `--vq-name` と一致させる |
| `:152`| `"split/version1/t2m_60_300/all.txt"` | `"split/romo_v1/t2m_60_300/all.txt"` |

### STEP 3: トークナイザ（§6 STEP A で詳述） → get_codes → T2M

学習3段のコマンドは §7 にまとめます。

---

## 7. 学習パイプライン（コマンド）

3段：**A. トークナイザ → B. コード化＋`all_data.pkl` → C. T2M（LLaMA）**。既存スクリプトの引数を差し替えます。

### STEP A: トークナイザ（`scripts/train/train_tokenizer.sh` を編集）

```bash
--dataname     motionmillion
--motion_type  vector_272_romo
--text_type    texts_romo
--version      romo_v1/tokenizer_96
--window-size  96
--nb-code      65536
--quantizer    FSQ
--exp-name     train_VQVAE_FSQ_romo272
# 選択肢：
#  (a) ゼロ学習: --resume-pth を付けない（最も安全、時間はかかる）
#  (b) 微調整:   --resume-pth checkpoints/pretrained_models/fsq_net_6000000.pth を付け少ない iter で
#      ← 272レイアウトが同一なので初期値として有効。RoMo 分布へ馴染ませる
```
```bash
bash scripts/train/train_tokenizer.sh
# 出力例: results/output/FSQ_96len/.../net_last.pth（次段で使う）
```

### STEP B: コード化＋`all_data.pkl`（`scripts/train/train_t2m_get_codes.sh` を編集）

```bash
--resume-pth   results/output/FSQ_96len/.../net_last.pth   # STEP A の成果物
--vq-name      VQVAE_codebook_65536_FSQ_romo               # STEP2 と一致
--dataname     motionmillion
--motion_type  vector_272_romo
--text_type    texts_romo
--version      romo_v1/t2m_60_300
```
```bash
bash scripts/train/train_t2m_get_codes.sh
# → dataset/MotionMillion/all_data.pkl と VQVAE_codebook_.../<id>.npy を生成
```

### STEP C: T2M 本体（`scripts/train/train_t2m_3B.sh` または `_7B.sh` を編集）

```bash
--resume-pth   results/output/FSQ_96len/.../net_last.pth   # STEP A のトークナイザ
--dataname     motionmillion
--motion_type  vector_272_romo
--text_type    texts_romo
--version      romo_v1/t2m_60_300
--train_split  train
--text_encode  flan-t5-xl        # checkpoints/models--google--flan-t5-xl（約43GB、DL済み）
```
```bash
bash scripts/train/train_t2m_3B.sh
```

---

## 8. 道(2)：RoMo-SMPL → `SMPLTo272Converter`（変換を自分で回す場合）

RoMo-272 を使わず SMPL から作りたい場合（SMPL も手元に残る）。`../motion-toolbox` を使います。

```python
from motion_toolbox.converters.motion_converters import SMPLTo272Converter
from pathlib import Path
import numpy as np

conv = SMPLTo272Converter(
    body_model_path=str(Path.home()/".motion_toolbox/datasets/body_models_clean_pkl"))

# RoMo-SMPL の各サンプル（dict: pose_body[T,63], root_orient[T,3], trans[T,3], betas[10]）を
smpl = {...}                       # ← RoMo-SMPL parquet の行から組み立てる
motion_272 = conv.convert(smpl)    # (T, 272) ＝ 本家 vector_272 と同一
np.save("dataset/MotionMillion/motion_data/vector_272_romo/train_000000.npy", motion_272)
```

- 以降のテキスト・split・mean/std・学習は §6/§7 と同じ。
- mean/std は **RoMo 全体から再算出**（RoMo-272 の同梱 Mean/Std は SMPL 経由では使えないので自前計算）：
  ```python
  import glob, numpy as np
  xs = [np.load(p) for p in glob.glob("dataset/MotionMillion/motion_data/vector_272_romo/*.npy")]
  allx = np.concatenate(xs,0); mean=allx.mean(0); std=allx.std(0); std[std<1e-6]=1e-6
  np.save(".../mean_std/vector_272_romo/mean.npy", mean); np.save(".../std.npy", std)
  ```

---

## 9. FPS とクリップ長（変換不要な点／要注意な点）

- **FPS 一致（30）**：`dataset_tokenize.py` の `motionmillion` 分岐は `fps=30`、RoMo も 30fps。リサンプリング不要。
  （`dataset_TM_train_motionmillion.py` の `fps=20` はトークン列段の記述で実フレームレートに無関係。）
- **クリップ長**：tokenizer は window=96 固定クロップ、T2M は `max_motion_length=300`（`block-size 301`）。
  96 未満は tokenizer split から除外（`dataset_VQ.py` が skip）。
  300 超を丸ごと学習したい場合のみ `max_motion_length` と位置エンコーディング長の見直しが必要。

---

## 10. ★現在足りないもの（前版から大幅に減少）

| # | 不足物 | 状態 | 対応 |
|---|--------|------|------|
| M1 | Parquet→`<id>.npy`+`texts`+`split` 展開スクリプト | 要自作（小） | §6 STEP1 |
| M2 | `datasets`/`pyarrow`/`huggingface_hub` | 未導入 | §6 STEP0 で pip |
| M3 | RoMo データ本体 | 未DL | §6 STEP0。まず小サブセット |
| M4 | 272 の可視化 | **✅ 解決済み** | `../motion-toolbox` の `Visualizer` / `Format272ToMotionConverter` |
| M5 | SMPL→`vector_272` 変換 | **✅ 解決済み** | `../motion-toolbox` の `SMPLTo272Converter`（往復検証済み）。道(1)なら変換自体不要 |
| M6 | `get_codes` 等のハードコードパス | 要修正（軽微） | §6 STEP2 |
| M7 | RoMo 分布のトークナイザ | 任意 | 272が同一なので事前学習を**初期値に微調整可**（§7 STEP A(b)）。厳密には再学習/微調整推奨 |
| M8 | motion-toolbox のセットアップ（道2/可視化時） | 要 | §3.1（`uv sync` ＋ SMPL body model） |
| M9 | RoMo 分布の評価器 | 任意 | 生成モデル学習には不要。厳密 FID を測るなら再学習 |

> 要するに、前版で「最大の壁」とした **M4（可視化）と M5（SMPL→272）は motion-toolbox が既に解決済み**。
> 残るのは「Parquet を展開する小スクリプト（M1）」「依存導入（M2/M3/M8）」「パス微修正（M6）」だけです。

---

## 11. 作業チェックリスト

**共通**
- [ ] `datasets`/`pyarrow`/`huggingface_hub` を学習環境に導入（M2）
- [ ] `../motion-toolbox` を `uv sync`／`pip install -e`（可視化・道2に使用、M8）
- [ ] 対象データセットの `column_names` を実データで確認

**道(1) RoMo-272 直接（推奨）**
- [ ] `RoMo-272` を DL し、§6 STEP1 で `vector_272_romo/`＋`texts_romo/`(5行)＋split を生成（M1）
- [ ] 同梱 `Mean.npy`/`Std.npy` を `mean_std/vector_272_romo/{mean,std}.npy` に配置
- [ ] **§6 STEP1.5 の1サンプル可視化でレイアウト一致を確認**（必須）
- [ ] §6 STEP2 の get_codes パス修正（M6）
- [ ] §7 STEP A→B→C（トークナイザ→コード化→T2M）を実行

**道(2) RoMo-SMPL 変換**
- [ ] SMPL body model を用意（§3.1）
- [ ] `SMPLTo272Converter` で `vector_272_romo/<id>.npy` を生成（§8）
- [ ] mean/std を RoMo 全体から再算出（§8）
- [ ] 以降は道(1)と同じ

---

## 12. 参考

- [`01_repository_workflow.md`](./01_repository_workflow.md) / [`02_paper_chapter4_architecture.md`](./02_paper_chapter4_architecture.md)
- RoMo データ: `https://huggingface.co/RoMoDataset`（SMPL / HML-263 / 272 / SOMA-77）
- RoMo 公式ツールキット: `../motion-toolbox`（`github.com/RoMoDataset/motion-toolbox`）
  - `docs/272_FORMAT_REFERENCE.md`（272 の定義）, `docs/272D_VALIDATION_RESULTS.md`（往復検証）
  - `src/motion_toolbox/converters/motion_converters.py`（`SMPLTo272Converter` ほか）
  - `src/motion_toolbox/visualization/`（`Visualizer`）, `scripts/convert_humanml3d_to_272.py`
- 272 参照実装: `github.com/Li-xingXiao/272-dim-Motion-Representation`（MotionMillion 共著者）
- 本リポジトリ根拠コード: `dataset/dataset_VQ.py`, `dataset/dataset_tokenize.py`,
  `train_t2m_get_codes.py`, `utils/motion_process.py`,
  `scripts/train/{train_tokenizer,train_t2m_get_codes,train_t2m_3B,train_t2m_7B}.sh`

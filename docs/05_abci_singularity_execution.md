# ABCI で Singularity 実行する手順（学習・推論コマンド集）

ABCI（PBS Professional / `qsub`）で本リポジトリを動かすための実践ガイドです。
ローカルの `docker-shell/run.sh` は **Docker 前提なので ABCI では使えません**。
ABCI では **ジョブスクリプト（本ドキュメントでは `pod.sh` と呼ぶ）を書き、その中で `singularity` を実行**します。

関連：[`04_apply_romo_dataset.md`](./04_apply_romo_dataset.md)（データ準備）、`docker-shell/Dockerfile`（イメージ定義）。

---

## 0. なぜ `run.sh` では動かないのか

| 項目 | ローカル `run.sh` | ABCI で必要な形 |
|------|-------------------|-----------------|
| コンテナ実行系 | `docker run` | `singularity exec/run`（`module load singularitypro`） |
| GPU 指定 | `--gpus all` | `--nv` |
| マウント | `-v host:cont` | `-B host:cont`（or `--bind`） |
| 実行方法 | ホストで直接 | **`qsub pod.sh`** でジョブ投入 |
| 権限 | root | 実行ユーザ（rootless）、**SIF は読み取り専用** |
| デーモン | Docker デーモン常駐 | 無し（デーモンレス） |

→ ABCI では「**イメージを SIF に変換 → リポジトリを `-B` でマウント → `pod.sh` から `singularity` 実行**」に置き換えます。

---

## 1. 全体の流れ

```
[手元 or ABCI] Dockerイメージ → SIF に変換（motionmillion.sif）
      │
[ABCI] リポジトリ一式（コード / checkpoints / dataset）をグループ領域へ配置
      │
[ABCI] pod.sh を qsub → ジョブ内で singularity exec --nv -B repo:/workspace ...
```

**重要**：`docker-shell/Dockerfile` は **コードを焼き込まず `requirements.txt` だけ COPY** します
（依存のみをイメージ化し、コード・モデル・データはマウントで渡す設計）。
したがって ABCI でも **リポジトリ全体を `/workspace` にバインド**して実行します。

---

## 2. 事前準備

### 2.1 SIF イメージを作る（3通り／推奨は A か B）

**A. Docker Hub 経由（最も安定・ABCI公式ハンズオン方式）**
```bash
# 手元（Dockerが使える環境）で
docker build -f docker-shell/Dockerfile -t <dockerhub_user>/motionmillion:cu118 .
docker push <dockerhub_user>/motionmillion:cu118

# ABCI のインタラクティブノードで
module load singularitypro/4.1.7
singularity build motionmillion.sif docker://<dockerhub_user>/motionmillion:cu118
```

**B. docker-archive 経由（Docker Hub を使わずファイル転送）**
```bash
# 手元で
docker build -f docker-shell/Dockerfile -t motionmillion:cu118 .
docker save motionmillion:cu118 -o motionmillion_docker.tar
# → motionmillion_docker.tar を ABCI へ scp/sftp 転送

# ABCI で
module load singularitypro/4.1.7
singularity build motionmillion.sif docker-archive://motionmillion_docker.tar
```

**C. spython で def 変換（非推奨・失敗しやすい）**
```bash
pip install spython
spython recipe docker-shell/Dockerfile motionmillion.def
singularity build --fakeroot motionmillion.sif motionmillion.def
```

> ⚠️ **CUDA 11.8 と ABCI 3.0（H200 = sm_90）の互換性**：本 Dockerfile は `cuda:11.8` ベース。
> CUDA 11.8 は Hopper(sm_90) に一応対応し、`--nv` でホストの新しいドライバが渡るため通常は動きますが、
> **必ず最初に小さなジョブで `torch.cuda.is_available()` と実演算を検証**してください。
> 問題が出たら Dockerfile のベースを `cuda:12.1`〜`12.4` 系＋`torch==2.4.1+cu121` 等へ差し替えます。

### 2.2 ディレクトリ配置（ABCI グループ領域）

```
/home/<user>/MotionMillion-Codes/     # ← リポジトリ一式（=$PBS_O_WORKDIR）
├── motionmillion.sif                 # 2.1 で作成
├── checkpoints/                      # 事前学習・flan-t5-xl 等（マウントで渡る）
├── dataset/                          # 学習データ
├── results/                          # 出力（ジョブが書き込む）
├── scripts/ ...                      # 学習/推論スクリプト
└── pod_*.sh                          # 本ドキュメントのジョブスクリプト
```

**pod.sh はリポジトリ直下から `qsub` する**運用にすると、`$PBS_O_WORKDIR` がリポジトリroot＝バインド元になり、
スクリプト内の相対パス（`results/output`, `checkpoints/...`, `dataset/...`）がそのまま解決できます。

### 2.3 共通の singularity 実行形（全 pod.sh で使い回す）

```bash
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"
# 使い方: $RUN bash scripts/....sh   /   $RUN python xxx.py
```

- `--nv`：GPU（必須）。`--cleanenv`：ホスト環境変数の汚染を遮断（推奨）。
- `-B $PBS_O_WORKDIR:/workspace --pwd /workspace`：リポジトリを丸ごと `/workspace` に割り当て。
- **書き込みは `/workspace` 配下（＝マウント先）のみ可**。SIF 内へは書けない。HuggingFace キャッシュは
  `checkpoints/...` に `local_files_only=True` で同梱済みなので実行時ダウンロードは不要。

---

## 3. pod.sh テンプレート集

PBS ヘッダの意味（ABCI 3.0）：

```bash
#PBS -P <group_id>          # 課金グループ（必須, 例: gcaXXXXX）
#PBS -q rt_HG               # 資源タイプ: rt_HF=ノード占有(8GPU) / rt_HG=共有(1GPU) / rt_HC=CPU
#PBS -l select=1:ncpus=16   # ノード数:CPUコア数（GPU数は資源タイプで決まる）
#PBS -l walltime=12:00:00   # 上限: On-demand 12h / Spot 168h
#PBS -N mm_job
#PBS -j oe                  # 標準出力/エラーを1ファイルに
```

> **num-workers を ncpus に合わせる**：リポジトリの学習スクリプトは `--num-workers 64` 等ですが、
> 共有ノード（rt_HG）では割当 CPU が少ないため、`ncpus` に応じて `--num-workers 8`〜`16` に**必ず下げて**ください
> （超過するとメモリ/CPU 逼迫でジョブが不安定化）。

### 3.1 推論（バッチ）★ABCI で最初に試すならこれ

`inference_batch.py` は `assets/infer_batch_prompt.txt` からプロンプトを読むので**バッチジョブ向き**。

`pod_infer_batch.sh`:
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=1:00:00
#PBS -N mm_infer_batch
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

$RUN bash scripts/inference/batch_inference/test_t2m_3B.sh
# 出力: results/output/inference/batch_inference/
```

### 3.2 推論（単一テキスト）— stdin に注意

`inference_single.py` は `input('Input text: ')` で**対話入力**します。バッチジョブには stdin が無いので、
**プロンプトをパイプで渡す**か、インタラクティブノードで実行します。

`pod_infer_single.sh`（プロンプトをパイプ）:
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=1:00:00
#PBS -N mm_infer_single
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

# "a person is walking" を1件生成（複数行流せば複数件）
printf 'a person is walking forward and then sits down.\n' | \
  $RUN bash scripts/inference/single_inference/test_t2m_3B.sh
# 出力: results/output/inference/single_inference/
```

> インタラクティブに試すなら：
> ```bash
> qrsh -P <group_id> -q rt_HG -l select=1:ncpus=16 -l walltime=1:00:00
> module load singularitypro/4.1.7
> singularity exec --nv --cleanenv -B $PWD:/workspace --pwd /workspace motionmillion.sif \
>   bash scripts/inference/single_inference/test_t2m_3B.sh
> ```

### 3.3 評価（tokenizer / T2M）

`pod_eval.sh`:
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=4:00:00
#PBS -N mm_eval
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

# どちらか
$RUN bash scripts/eval/eval_tokenizer.sh      # トークナイザ再構成評価
# $RUN bash scripts/eval/eval_t2m_3B.sh       # T2M 評価（要 --resume-trans の学習済みパス）
```

### 3.4 トークナイザ学習（単一ノード）

**1GPU（rt_HG）**：
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=168:00:00
#PBS -N mm_tok_1gpu
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

# num-workers を ncpus に合わせて下げること（スクリプトを編集 or 下記のように上書き実行）
$RUN bash scripts/train/train_tokenizer_single_gpu.sh
```

**8GPU（rt_HF＝ノード占有）**：`train_tokenizer.sh` は 4GPU 前提なので 8 に上書きします。
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=168:00:00
#PBS -N mm_tok_8gpu
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

$RUN bash -c '
export NCCL_TIMEOUT=1200
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --main_process_port 29701 train_tokenizer.py \
  --batch-size 64 --lr 5e-5 --total-iter 6000000 --lr-scheduler 300000 \
  --down-t 1 --depth 3 --dilation-growth-rate 3 --out-dir results/output/FSQ_96len \
  --dataname motionmillion --vq-act relu --quantizer FSQ --loss-vel 0.5 --recons-loss l1_smooth \
  --exp-name train_VQVAE_FSQ_bz64_codebook65536_motionmillion_8gpu_96window \
  --nb-code 65536 --motion_type vector_272 --version version1/tokenizer_96 \
  --warm-up-iter 48000 --num-workers 16 --window-size 96 --kernel-size 3 \
  --use_patcher --patch_size 1 --patch_method haar --vq-norm LN
'
```

### 3.5 コード化（get_codes）＋ `all_data.pkl`（1GPU）

```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=24:00:00
#PBS -N mm_getcodes
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

# train_t2m_get_codes.sh 内の --resume-pth を学習済みトークナイザに合わせてから
$RUN bash scripts/train/train_t2m_get_codes.sh
```

### 3.6 T2M 学習（3B / 7B）— 単一ノード8GPU（推奨・現実的）

元の `train_t2m_3B.sh` は **5ノード×8GPU=40プロセス**のマルチノード設定ですが、
ABCI 3.0 の 1 ノード（8×H200, 各141GB）なら**単一ノード8GPUで回せます**。
launcher を単一ノード用に上書きし、学習引数は元スクリプトと同一にします。

`pod_t2m_3B_1node.sh`:
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=168:00:00
#PBS -N mm_t2m_3B
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"
RUN="singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF"

$RUN bash -c '
export NCCL_TIMEOUT=1200
accelerate launch --num_processes 8 --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port 29701 train_t2m_llama.py \
  --exp-name train_mm_GPT_65536_FSQ_llama_3B_t5xl_bf16_1node8gpu \
  --batch-size 16 --num-layers 9 --embed-dim-gpt 1024 --nb-code 65536 --n-head-gpt 16 \
  --block-size 301 --ff-rate 4 --drop-out-rate 0.1 \
  --resume-pth ./checkpoints/pretrained_models/fsq_net_6000000.pth \
  --vq-name VQVAE_codebook_65536_FSQ_all --out-dir results/output/T2M/600iterFSQ \
  --total-iter 240000 --lr-scheduler-type CosineDecayScheduler --lr 0.0004 \
  --dataname motionmillion --down-t 1 --depth 3 --quantizer FSQ --dilation-growth-rate 3 \
  --vq-act relu --vq-norm LN --fps 30 --kernel-size 3 --use_patcher --patch_size 1 --patch_method haar \
  --text_encode flan-t5-xl --pretrained_llama 3B --pkeep 1 \
  --motion_type vector_272 --text_type texts --version version1/t2m_60_300 \
  --mixed_precision bf16 --save-iter-last 1000 --gradient_accumulation_steps 1 \
  --save-iter 10000 --train_split train
'
```

**7B の場合**：`--pretrained_llama 7B`、`--exp-name` を 7B 用に変更。VRAM が厳しければ DeepSpeed ZeRO-2 を使用：
`accelerate launch --config_file configs/accelerate_configs/mnode_deepspeed_zero2.yaml --num_processes 8 --num_machines 1 --machine_rank 0 --main_process_ip 127.0.0.1 --main_process_port 29701 train_t2m_llama.py ...`（以降の引数は上記と同様、`--pretrained_llama 7B`）。

### 3.7 T2M 学習 — マルチノード（上級・元設定の 5ノード×8GPU=40）

複数ノード予約時のテンプレート。PBS の `$PBS_NODEFILE` からノード一覧を取り、各ノードで `machine_rank` を振って起動します。

`pod_t2m_3B_multinode.sh`:
```bash
#!/bin/bash
#PBS -P <group_id>
#PBS -q rt_HF
#PBS -l select=5                 # 5ノード（=40GPU）
#PBS -l walltime=168:00:00
#PBS -N mm_t2m_3B_mn
#PBS -j oe

cd $PBS_O_WORKDIR
module load singularitypro/4.1.7
SIF="$PBS_O_WORKDIR/motionmillion.sif"

NODES=($(sort -u "$PBS_NODEFILE"))
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29701

for i in "${!NODES[@]}"; do
  ssh -o StrictHostKeyChecking=no "${NODES[$i]}" bash -lc "'
    cd $PBS_O_WORKDIR && module load singularitypro/4.1.7 && \
    MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$i WORLD_SIZE=$NNODES \
    singularity exec --nv --cleanenv -B $PBS_O_WORKDIR:/workspace --pwd /workspace $SIF \
    bash -c \"export NCCL_TIMEOUT=1200; \
      accelerate launch --config_file configs/accelerate_configs/mnode8gpu.yaml \
        --main_process_ip=\$MASTER_ADDR --main_process_port=\$MASTER_PORT \
        --machine_rank=\$RANK --num_machines=\$WORLD_SIZE --num_processes=\$((WORLD_SIZE*8)) \
        train_t2m_llama.py <train_t2m_3B.sh と同じ引数群> \"
  '" &
done
wait
```

> マルチノードは **NCCL の InfiniBand 設定**（`NCCL_IB_*`）や ssh 到達性など環境依存要因が多く、
> まず 3.6 の単一ノードで確実に回してから移行することを強く推奨します。

---

## 4. 実行コマンド早見表（コンテナ内で走る中身）

| 目的 | スクリプト / コマンド | 並列 | 資源タイプ |
|------|----------------------|------|-----------|
| バッチ推論 | `bash scripts/inference/batch_inference/test_t2m_3B.sh` | 1GPU | rt_HG |
| 単一推論 | `... | bash scripts/inference/single_inference/test_t2m_3B.sh`（stdin） | 1GPU | rt_HG |
| tokenizer 評価 | `bash scripts/eval/eval_tokenizer.sh` | 1GPU | rt_HG |
| T2M 評価 | `bash scripts/eval/eval_t2m_3B.sh` | 1GPU | rt_HG |
| tokenizer 学習(1GPU) | `bash scripts/train/train_tokenizer_single_gpu.sh` | 1GPU | rt_HG |
| tokenizer 学習(8GPU) | `accelerate launch --num_processes 8 ... train_tokenizer.py`（§3.4） | 8GPU | rt_HF |
| get_codes | `bash scripts/train/train_t2m_get_codes.sh` | 1GPU | rt_HG |
| T2M 3B 学習 | `accelerate launch --num_processes 8 --num_machines 1 ... train_t2m_llama.py`（§3.6） | 8GPU | rt_HF |
| T2M 7B 学習 | 同上＋`--pretrained_llama 7B`＋DeepSpeed ZeRO-2 | 8GPU〜 | rt_HF |

### 標準的な学習フロー（RoMo/MotionMillion 共通）
1. **tokenizer 学習**（§3.4）→ `results/output/FSQ_96len/.../net_*.pth`
2. **get_codes**（§3.5、`--resume-pth` を 1 の成果物に）→ コード＋`all_data.pkl`
3. **T2M 学習**（§3.6、`--resume-pth` を 1 の成果物に）→ `results/output/T2M/...`
4. **評価/推論**（§3.1〜3.3）

---

## 5. ジョブ投入・監視・落とし穴

### 5.1 投入と監視
```bash
qsub pod_infer_batch.sh        # 投入 → ジョブID が返る
qstat                          # 自分のジョブ状態
qstat -f <jobid>               # 詳細
qdel <jobid>                   # キャンセル
cat mm_infer_batch.o<jobid>    # 標準出力（-j oe で統合）
```

### 5.2 よくある落とし穴

| 症状 | 原因 | 対処 |
|------|------|------|
| `torch.cuda.is_available()==False` | `--nv` 忘れ | singularity に `--nv` |
| コードが見つからない / 相対パス失敗 | バインド or `--pwd` 忘れ | `-B $PBS_O_WORKDIR:/workspace --pwd /workspace` |
| 書き込みエラー（Read-only file system） | SIF 内へ書こうとした | 出力先を `/workspace` 配下に。追加 pip は不可 |
| 単一推論が固まる / EOFError | `input()` に stdin 無し | §3.2 のパイプ、or インタラクティブノード |
| DataLoader で Bus error / OOM | `--num-workers` 過大・shm 不足 | ncpus に合わせ `--num-workers` を下げる |
| H200 で CUDA エラー | cu118 と sm_90 の相性 | §2.1 の検証。必要なら cu121/cu124 ベースへ |
| ポート衝突（複数ジョブ同居） | `--main_process_port` 固定 | ジョブごとに別ポート（29701, 29702, ...） |
| ホスト環境変数の混入 | env 継承 | `--cleanenv`（必要なら `--env VAR=...` で明示注入） |
| flan-t5-xl 読込失敗 | パス不一致 | `checkpoints/models--google--flan-t5-xl/snapshots/<hash>/` が存在するか確認（`local_files_only=True` のためDL不可） |

### 5.3 メモ
- **walltime 上限**：On-demand 12h / Spot 168h。長時間学習は Spot＋`--save-iter` によるチェックポイントで途中再開（`--resume-trans`）。
- **num_processes は「割当GPU数」に一致**させる（単一ノードは 8、`num_machines 1`）。
- **checkpoints/dataset は巨大**（flan-t5-xl 43GB 等）。グループ領域の容量とクォータに注意。

---

## 6. 参考
- ABCI ハンズオン例：`../abci-ws-hands-on/job.sh`, `README.md`（PBS/singularity の最小例）
- 本リポジトリ：`docker-shell/Dockerfile`（イメージ定義）、`scripts/train`・`scripts/eval`・`scripts/inference`
- accelerate 設定：`configs/accelerate_configs/{mnode8gpu,mnode_deepspeed_zero2}.yaml`
- データ準備（RoMo 適用）：[`04_apply_romo_dataset.md`](./04_apply_romo_dataset.md)

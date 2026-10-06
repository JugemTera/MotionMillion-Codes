#!/bin/bash
# =============================================================================
# MotionMillion データセットのセットアップ（ダウンロード → 展開 → 配置 → 検証）
#
#   HF: https://huggingface.co/datasets/InternRobotics/MotionMillion  (gated)
#
# 前提:
#   * HF 上でライセンスに同意済みで、トークンが使えること
#       - 環境変数 HF_TOKEN を設定する   または
#       - `huggingface-cli login` 済み (~/.cache/huggingface/token)
#   * 空きディスク: 圧縮 tar.gz 約 320 GB (Mirror 込み / 除外時 約 160 GB)
#                  + 展開後の .npy (圧縮の 1.5〜2 倍程度を見込む)
#
# 使い方 (リポジトリルートから):
#   bash prepare/setup_motionmillion_dataset.sh            # 全工程
#   STAGES=download bash prepare/setup_motionmillion_dataset.sh   # 工程を選んで実行
#
# 環境変数:
#   STAGES          実行する工程。download,extract,arrange,verify  [default: all]
#                   区切りは , : + のどれでも可 (qsub -v で渡すときは STAGES=arrange+verify)
#   DATA_ROOT       最終配置先                      [default: dataset/MotionMillion]
#   RAW_DIR         tar.gz の置き場 (別領域に逃がせる) [default: $DATA_ROOT/raw_hf]
#   INCLUDE_MIRROR  1=Mirror_* も取得, 0=除外       [default: 1]
#   SUBSETS         取得する motion サブセット (スペース区切り)
#                   [default: MotionGV MotionLLAMA MotionUnion PhantomDanceDatav1.1]
#   EXTRACT_JOBS    tar 展開の並列数                 [default: nproc, 最大 16]
#   HF_MAX_WORKERS  HF ダウンロードの並列数          [default: 8]
#
# 全工程とも再実行可能(resumable)です:
#   download : huggingface-cli が未完了ファイルだけ再取得
#   extract  : <archive>.done マーカーがあるものはスキップ
#   arrange  : 既に正しい場所にあるファイルはスキップ (hardlink で配置)
# =============================================================================
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

HF_REPO="InternRobotics/MotionMillion"
DATA_ROOT=${DATA_ROOT:-dataset/MotionMillion}
RAW_DIR=${RAW_DIR:-$DATA_ROOT/raw_hf}
STAGING_DIR=$DATA_ROOT/_extracted
STAGES=${STAGES:-all}
INCLUDE_MIRROR=${INCLUDE_MIRROR:-1}
SUBSETS=${SUBSETS:-"MotionGV MotionLLAMA MotionUnion PhantomDanceDatav1.1"}
EXTRACT_JOBS=${EXTRACT_JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-8}

log() { echo "[$(date '+%F %T')] $*"; }
# STAGES の区切りは , : + 空白 のいずれでも可 (qsub -v はカンマを変数区切りに使うため)
run_stage() { local st=",${STAGES//[:+ ]/,},"; [[ "$STAGES" == all || "$st" == *",$1,"* ]]; }

mkdir -p "$DATA_ROOT" "$RAW_DIR" "$STAGING_DIR"

# -----------------------------------------------------------------------------
# 0. 取得対象パターンの組み立て
# -----------------------------------------------------------------------------
# NOTE: huggingface-cli の --include は nargs="*" なので、--include を複数回書くと
#       最後の 1 つしか効かない。パターンは 1 つの --include の後ろにまとめて渡す。
INCLUDE_PATTERNS=("split.tar.gz" "texts.tar.gz" "mean_std/*" "README.md")
for s in $SUBSETS; do
  if [[ "$s" == "PhantomDanceDatav1.1" ]]; then
    INCLUDE_PATTERNS+=("motion_272rpr/${s}.tar.gz")
    [[ "$INCLUDE_MIRROR" == 1 ]] && INCLUDE_PATTERNS+=("motion_272rpr/Mirror_${s}.tar.gz")
  else
    INCLUDE_PATTERNS+=("motion_272rpr/${s}/*.tar.gz")
    [[ "$INCLUDE_MIRROR" == 1 ]] && INCLUDE_PATTERNS+=("motion_272rpr/Mirror_${s}/*.tar.gz")
  fi
done

# -----------------------------------------------------------------------------
# 1. download
# -----------------------------------------------------------------------------
if run_stage download; then
  log "=== [download] $HF_REPO -> $RAW_DIR"

  if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN が未設定です。gated データセットのため、HF でライセンスに同意した上で" >&2
    echo "       export HF_TOKEN=hf_xxx  か  huggingface-cli login  を行ってください。" >&2
    exit 1
  fi
  export HF_HUB_DISABLE_TELEMETRY=1
  # メタデータキャッシュを RAW_DIR 側に置き、$HOME を圧迫しない
  export HF_HOME=${HF_HOME:-$RAW_DIR/.hf_home}

  if ! command -v huggingface-cli >/dev/null 2>&1; then
    log "huggingface-cli が見つからないため pip install --user します"
    pip install --user --no-cache-dir "huggingface_hub>=0.30"
    export PATH="$HOME/.local/bin:$PATH"
  fi

  log "include patterns: ${INCLUDE_PATTERNS[*]}"
  huggingface-cli download "$HF_REPO" \
    --repo-type dataset \
    --local-dir "$RAW_DIR" \
    --max-workers "$HF_MAX_WORKERS" \
    --include "${INCLUDE_PATTERNS[@]}"

  log "[download] done. size:"
  du -sh "$RAW_DIR" || true
  n_tar=$(find "$RAW_DIR" -name '*.tar.gz' -not -path '*/.cache/*' | wc -l)
  log "[download] tar.gz files: $n_tar"
  for must in split.tar.gz texts.tar.gz; do
    if [[ ! -f "$RAW_DIR/$must" ]]; then
      echo "ERROR: $RAW_DIR/$must がありません。ダウンロードが不完全です (上の include patterns と HF 側の承認状態を確認)。" >&2
      exit 1
    fi
  done
fi

# -----------------------------------------------------------------------------
# 2. extract
#    各 tar.gz を STAGING_DIR/<archive 相対パス(拡張子なし)>/ に展開する。
#    tar 内部のディレクトリ構成が不明なため、ここでは一切リネームせず
#    archive ごとに隔離して展開し、配置は arrange 工程 (Python) に任せる。
# -----------------------------------------------------------------------------
if run_stage extract; then
  log "=== [extract] $RAW_DIR -> $STAGING_DIR  (jobs=$EXTRACT_JOBS)"

  if command -v pigz >/dev/null 2>&1; then DECOMP="pigz -dc"; else DECOMP="gzip -dc"; fi
  export DECOMP STAGING_DIR RAW_DIR

  extract_one() {
    local archive="$1"                              # RAW_DIR からの相対パス
    local stem="${archive%.tar.gz}"
    local dest="$STAGING_DIR/$stem"
    local marker="$STAGING_DIR/$stem.done"
    if [[ -f "$marker" ]]; then
      echo "skip (done): $archive"; return 0
    fi
    rm -rf "$dest"; mkdir -p "$dest"
    echo "extracting: $archive"
    if $DECOMP "$RAW_DIR/$archive" | tar -x -C "$dest"; then
      touch "$marker"
    else
      echo "FAILED: $archive" >&2; rm -rf "$dest"; return 1
    fi
  }
  export -f extract_one

  ( cd "$RAW_DIR" && find . -name '*.tar.gz' -type f -not -path '*/.cache/*' | sed 's#^\./##' | sort ) > "$STAGING_DIR/.archives.txt"
  log "$(wc -l < "$STAGING_DIR/.archives.txt") archives"
  if [[ ! -s "$STAGING_DIR/.archives.txt" ]]; then
    echo "ERROR: $RAW_DIR に tar.gz がありません。先に download 工程を実行してください。" >&2
    exit 1
  fi

  # split / texts は小さいので先に、motion は並列で
  # (grep は一致 0 件で exit 1 を返すため、pipefail で落ちないよう || true を付ける)
  { grep -E '^(split|texts)\.tar\.gz$' "$STAGING_DIR/.archives.txt" || true; } \
    | while read -r a; do extract_one "$a"; done
  { grep -vE '^(split|texts)\.tar\.gz$' "$STAGING_DIR/.archives.txt" || true; } \
    | xargs -r -P "$EXTRACT_JOBS" -I{} bash -c 'extract_one "$1"' _ {}

  n_total=$(wc -l < "$STAGING_DIR/.archives.txt")
  n_done=$(find "$STAGING_DIR" -name '*.done' | wc -l)
  log "[extract] done markers: $n_done / $n_total"
  if [[ "$n_done" -lt "$n_total" ]]; then
    echo "ERROR: 展開に失敗した archive があります。再実行すると未完了分だけやり直します。" >&2
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 3. arrange
#    STAGING_DIR 以下の .npy / .txt を split の名前に突き合わせ、
#      $DATA_ROOT/motion_data/vector_272/<name>.npy
#      $DATA_ROOT/texts/<name>.txt
#      $DATA_ROOT/split/version1/...
#      $DATA_ROOT/mean_std/vector_272/{mean,std}.npy
#    を作る (hardlink)。見つからない名前を除外した split を
#      $DATA_ROOT/split/version1_avail/...
#    にも書き出す。
# -----------------------------------------------------------------------------
if run_stage arrange; then
  log "=== [arrange] staging -> $DATA_ROOT"
  python prepare/arrange_motionmillion.py \
    --data-root "$DATA_ROOT" \
    --staging-dir "$STAGING_DIR" \
    --raw-dir "$RAW_DIR"
fi

# -----------------------------------------------------------------------------
# 4. verify  (学習時のロードを模した軽い検査)
# -----------------------------------------------------------------------------
if run_stage verify; then
  log "=== [verify]"
  python prepare/arrange_motionmillion.py --data-root "$DATA_ROOT" --staging-dir "$STAGING_DIR" --verify-only
fi

log "all requested stages finished: $STAGES"

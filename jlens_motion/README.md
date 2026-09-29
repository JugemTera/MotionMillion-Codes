# jlens_motion ―― MotionMillion 用 Jacobian lens（ヤコビアン計算部）

[Jacobian lens](https://transformer-circuits.pub/2026/workspace/)（公式実装 [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)）を
`models/lit_llama/model_hf.py` の `LLaMAHF` に適用するためのパッケージ。
背景と検証計画は `docs/06_jlens_circuit_analysis.md` と `docs/07_jlens_verification_plan.md` を参照。

現時点で実装しているのは **ヤコビアン $J_\ell$ の計算（fit）と、それを使った読み出し** まで。
読み出し結果の解析（FSQ デコード、6 軸への射影、steering）はまだ無い。

## 何を計算するか

層 $\ell$ の残差 $h_{\ell,p}$（位置 $p$）から最終ブロック出力 $h_{final,q}$ への平均ヤコビアン：

$$
J_\ell = \mathbb{E}_{\text{example}}\ \mathbb{E}_{p \in S}\ \sum_{q \in T} \frac{\partial h_{final,q}}{\partial h_{\ell,p}}
\qquad(\text{$d \times d$、$d$ = n\_embd})
$$

推定量は公式と同じ（出力次元ごとに one-hot を全ターゲット位置へ同時に流して逆伝播し、ソース位置で平均）。
読み出しは $\mathrm{lens}_\ell(h) = \texttt{lm\_head}(\texttt{ln\_f}(J_\ell h))$ で、語彙は **65,536 個の FSQ モーションコード＋終了トークン**。

言語モデル版との違いは **ソース位置 $S$ とターゲット位置 $T$ の選び方**だけ：

| モード | $S$ | $T$ | 問い |
|--------|-----|-----|------|
| `motion`（M-lens） | モーション位置 | モーション位置 | 層 $\ell$ はどんな動きを出そうとしているか |
| `text_to_motion` | テキスト位置 | モーション位置 | このプロンプトは何を生成させようとしているか |
| `offset` | モーション位置 $q-\Delta$ | モーション位置 $q$ | 層 $\ell$ は何ステップ先まで決めているか（$J_\ell^{(\Delta)}$） |

- 系列は `[テキスト (Lt) | c_0 … c_{N-1}]`。終了トークンは付けない。位置 $p$ の残差は $p+1$ のトークンを予測する。
- テキスト部は双方向注意なので、**テキスト位置はターゲットにしない**（公式の推定量は因果構造を前提にしている）。
- 公式の「先頭 16 位置を除外」は言語モデルの attention sink 対策なので使わない。代わりに `--skip-first` でモーション先頭を除外できる。
- 平均した $J_\ell$ は将来位置を混ぜるので「何ステップ先か」は分からない。それを測るのが `offset` モード（1 つのターゲット $q$ ごとに逆伝播すると、全ソース位置＝全 $\Delta$ の勾配が一度に得られる）。

## コスト

| モード | 1 例あたりの backward 回数 | 保存サイズ |
|--------|---------------------------|------------|
| `motion` / `text_to_motion` | $\lceil d / \text{dim\_batch} \rceil$ | 層数 × $d^2$ × 2 byte（fp16 保存） |
| `offset` | ターゲット数 × $\lceil d / \text{dim\_batch} \rceil$ | 層数 × オフセット数 × $d^2$ × 4 byte |

3B（24 層、$d=3200$）で `dim_batch=8` なら 1 例 400 回。全層の $J$ は約 0.5 GB（fp16）。
`dim_batch` を上げるとメモリと引き換えに回数が減るが、backward の総 FLOPs は変わらない。実時間はまず数例で測ること。

## 使い方

```bash
# M-lens（全層、100 例、途中再開可能）
python fit_jlens.py --pretrained_llama 3B \
  --resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth \
  --mode motion --n-examples 100 \
  --out results/jlens/3B_motion.pt --checkpoint results/jlens/3B_motion.ckpt

# オフセット別（層とオフセットを絞る）
python fit_jlens.py ... --mode offset --layers 6,12,18 --offsets 0,1,2,4,8,16 \
  --out results/jlens/3B_offset.pt
```

データは学習時と同じ `dataset/MotionMillion/all_data.pkl`（`code_data` / `text_data`）と split ファイルから、`--seed` で決定的にサンプルする。
キャプションは flan-t5-xl（`--t5-path`）で先にエンコードし、T5 を解放してからフィットする。
ABCI 用のひな形は `scripts/jlens/fit_jlens_3B.sh`。

Python から：

```python
from jlens_motion import JacobianLens, MotionLensModel, lens_readout
lens = JacobianLens.load("results/jlens/3B_motion.pt")
model = MotionLensModel(net)                      # 読み込み済みの LLaMAHF
logit, final = lens_readout(model, example, layers=[6, 12, 18])              # logit lens（基準線）
jlens, final = lens_readout(model, example, layers=[6, 12, 18], lens=lens)   # J-lens
```

`JacobianLens.merge` で、別ジョブで fit した分割を例数の重み付き平均で結合できる。

## テスト

```bash
python -m pytest tests/test_jlens_motion.py
```

ランダム初期化の小さな `LLaMAHF`（CPU、重み不要）で以下を確認している。

- `MotionLensModel.forward` → `unembed` が `LLaMAHF.forward` と一致する
- 推定量が、下流ブロックを `torch.autograd.functional.jacobian` で直接微分した値と一致する（`motion` / `text_to_motion` / `offset`）
- `dim_batch` を変えても結果が変わらない
- モーション位置からテキスト位置へのヤコビアンが厳密に 0（テキストはモーションを見ない）
- 単位行列のレンズが logit lens と一致し、最終層の読み出しがモデル出力と一致する
- fit の途中再開・分割結合・保存/読み込み

## 未実装

- 読み出し結果の評価（最終出力トークンの順位・KL を層ごとに集計する検証 1・2 のスクリプト）
- 上位コードの FSQ デコード・6 軸射影・可視化
- steering（読み出し方向への介入）

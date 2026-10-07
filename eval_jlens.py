"""Compare a fitted Jacobian lens against the logit lens (docs/07 checks 1 and 2).

For held-out (caption, code sequence) pairs the model is run once with
teacher forcing; at every motion position and every layer the residual is
read out (a) directly, ``unembed(h_l)`` (logit lens), and (b) through the
fitted Jacobian, ``unembed(J_l h_l)`` (J-lens). Both readouts are compared
with the model's own final-layer prediction at the same position.

    python eval_jlens.py --pretrained_llama 3B \
        --resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth \
        --lens results/jlens/3B_all_val_motion_n1000.pt \
        --n-examples 300 --seed 1 --out results/jlens/eval_3B_all_val_motion_n1000

Outputs ``<out>.json`` (per-layer metrics plus per-example records),
``<out>.csv`` (per-layer table) and ``<out>.png`` (curves).

Metrics (averaged over all evaluated positions):
    top1        lens argmax == final argmax
    top10       final argmax inside the lens top-10
    gt_top1     lens argmax == the actual next code (teacher-forcing target)
    kl          KL(final || lens) in nats
    log10_rank  log10 of the rank of the final argmax under the lens (0 = top-1)
    repeat      lens argmax == the code at the current position
                ("keep doing the same thing" readout)
The final layer itself is reported as the reference row (J = I there).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch

from fit_jlens import DTYPES, encode_examples, load_model, sample_pairs
from jlens_motion import JacobianLens, lens_readout, motion_positions

logger = logging.getLogger("eval_jlens")

METRICS = ("top1", "top10", "gt_top1", "kl", "log10_rank", "repeat")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_llama", default="3B")
    parser.add_argument("--resume-trans", required=True)
    parser.add_argument("--nb-code", type=int, default=64000)
    parser.add_argument("--block-size", type=int, default=301)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--t5-path", default="checkpoints/flan-t5-xl")
    parser.add_argument("--clip-dim", type=int, default=2048)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="./dataset/MotionMillion")
    parser.add_argument("--split-file", default="./dataset/MotionMillion/split/version1_avail/t2m_60_300/val.txt")
    parser.add_argument("--n-examples", type=int, default=300)
    parser.add_argument("--max-text-length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1, help="sampling seed; use one that differs from the fit")
    parser.add_argument("--exclude-fit-seed", type=int, default=0, help="drop the examples a fit with this seed used (-1: keep)")
    parser.add_argument("--exclude-fit-n", type=int, default=1000)
    parser.add_argument("--lens", required=True, help="JacobianLens .pt from fit_jlens.py")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--future-offsets", default="0,1,2,3,4,6,8,12,16", help="also score each readout against the final output Δ positions ahead")
    parser.add_argument("--out", required=True, help="output prefix (.json/.csv/.png are appended)")
    return parser.parse_args()


def exclude_fit_examples(args) -> set[str]:
    """Names the fit sampled with ``(--exclude-fit-seed, --exclude-fit-n)``."""
    if args.exclude_fit_seed < 0:
        return set()
    fit_args = argparse.Namespace(**vars(args))
    fit_args.seed, fit_args.n_examples = args.exclude_fit_seed, args.exclude_fit_n
    return {name for name, _, _ in sample_pairs(fit_args)}


def sample_heldout(args) -> list[tuple[str, str, list[int]]]:
    used = exclude_fit_examples(args)
    wide = argparse.Namespace(**vars(args))
    wide.n_examples = args.n_examples + len(used)
    pairs = [p for p in sample_pairs(wide) if p[0] not in used][: args.n_examples]
    logger.info("sampled %d held-out examples (excluded %d fit examples)", len(pairs), len(used))
    return pairs


def joint_speed(args, name: str) -> float | None:
    """Mean joint speed (m/s) of the raw clip, for a static/dynamic breakdown."""
    path = os.path.join(args.data_root, "motion_data", "vector_272", name + ".npy")
    try:
        a = np.load(path)
    except OSError:
        return None
    vel = a[1:, 74:140].reshape(len(a) - 1, 22, 3)
    return float(np.linalg.norm(vel, axis=-1).mean() * 30)


def position_metrics(lens_logits: torch.Tensor, final_logits: torch.Tensor, gt: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    """All metrics for one layer, averaged over the positions (rows)."""
    final_arg = final_logits.argmax(-1)
    lens_arg = lens_logits.argmax(-1)
    final_score = lens_logits.gather(1, final_arg[:, None])
    rank = (lens_logits > final_score).sum(-1)  # 0 = top-1
    log_p_final = torch.log_softmax(final_logits, -1)
    log_p_lens = torch.log_softmax(lens_logits, -1)
    kl = (log_p_final.exp() * (log_p_final - log_p_lens)).sum(-1)
    return {
        "top1": (lens_arg == final_arg).float().mean().item(),
        "top10": (rank < 10).float().mean().item(),
        "gt_top1": (lens_arg == gt).float().mean().item(),
        "kl": kl.mean().item(),
        "log10_rank": torch.log10(rank.float() + 1).mean().item(),
        "repeat": (lens_arg == current).float().mean().item(),
    }


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    pairs = sample_heldout(args)
    examples = encode_examples(args, pairs)
    model = load_model(args)
    lens = JacobianLens.load(args.lens)
    logger.info("%s / %s", model, lens)
    final_layer = model.n_layers - 1
    layers = list(range(model.n_layers))  # the final layer is the identity reference

    sums = {kind: {l: {m: 0.0 for m in METRICS} for l in layers} for kind in ("logit", "jlens")}
    offsets = [int(x) for x in args.future_offsets.split(",")]
    fut = {kind: {l: {d: [0.0, 0.0, 0] for d in offsets} for l in layers} for kind in ("logit", "jlens")}  # top1 sum, kl sum, count
    n_pos_total = 0
    records = []
    t0 = time.time()
    for i, ((name, _, _), ex) in enumerate(zip(pairs, examples), 1):
        pos = motion_positions(ex, skip_first=args.skip_first)
        k = pos - ex.text_len  # position Lt+k holds c_k and predicts c_{k+1}
        current, gt = ex.codes[k], ex.codes[k + 1]
        per_ex = {"name": name, "caption": ex.caption, "n_pos": int(len(pos)), "joint_speed": joint_speed(args, name)}
        for kind, J in (("logit", None), ("jlens", lens)):
            lens_logits, final_logits = lens_readout(model, ex, layers, lens=J, positions=pos)
            per_ex[kind] = {}
            for l in layers:
                m = position_metrics(lens_logits[l], final_logits, gt, current)
                per_ex[kind][l] = m
                for key, val in m.items():
                    sums[kind][l][key] += val * len(pos)
                for d in offsets:  # readout at p scored against the final output at p+d
                    n = len(pos) - d
                    if n <= 0:
                        continue
                    f = future_metrics(lens_logits[l][:n], final_logits[d:])
                    fut[kind][l][d][0] += f["top1"] * n
                    fut[kind][l][d][1] += f["kl"] * n
                    fut[kind][l][d][2] += n
            per_ex["final_gt_top1"] = (final_logits.argmax(-1) == gt).float().mean().item()
        n_pos_total += len(pos)
        records.append(per_ex)
        if i % 10 == 0 or i == len(examples):
            j, g = per_ex["jlens"], per_ex["logit"]
            logger.info(
                "example %d/%d  pos=%d  %.0fs  top1@L12 logit=%.2f jlens=%.2f  @L18 logit=%.2f jlens=%.2f",
                i, len(examples), len(pos), time.time() - t0,
                g[12]["top1"], j[12]["top1"], g[18]["top1"], j[18]["top1"],
            )

    per_layer = {kind: {l: {m: sums[kind][l][m] / n_pos_total for m in METRICS} for l in layers} for kind in sums}
    per_future = {kind: {l: {d: {"top1": v[0] / max(v[2], 1), "kl": v[1] / max(v[2], 1), "n": v[2]} for d, v in fut[kind][l].items()} for l in layers} for kind in fut}
    out = {
        "per_future": per_future,
        "lens": args.lens, "lens_meta": {k: v for k, v in lens.meta.items() if k != "captions"},
        "n_examples": len(examples), "n_positions": n_pos_total, "seed": args.seed,
        "final_gt_top1": float(np.mean([r["final_gt_top1"] for r in records])),
        "per_layer": per_layer, "examples": records,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [f"{kind}_{m}" for kind in ("logit", "jlens") for m in METRICS])
        for l in layers:
            w.writerow([l] + [f"{per_layer[kind][l][m]:.4f}" for kind in ("logit", "jlens") for m in METRICS])
    plot(per_layer, layers, final_layer, args.out + ".png")
    plot_future(per_future, offsets, args.out + "_future.png")
    logger.info("top-1 agreement with the final output Δ positions ahead (logit / jlens)")
    for l in (12, 16, 18, 20, 22):
        if l in per_future["logit"]:
            logger.info("L%-2d " + "  ".join(f"Δ{d}: %.3f/%.3f" for d in offsets), l, *[x for d in offsets for x in (per_future["logit"][l][d]["top1"], per_future["jlens"][l][d]["top1"])])

    logger.info("layer  logit top1  jlens top1 | logit top10  jlens top10 | logit KL  jlens KL | repeat logit/jlens")
    for l in layers:
        g, j = per_layer["logit"][l], per_layer["jlens"][l]
        logger.info("%5d  %10.3f  %10.3f | %11.3f  %11.3f | %8.2f  %8.2f | %.2f / %.2f",
                    l, g["top1"], j["top1"], g["top10"], j["top10"], g["kl"], j["kl"], g["repeat"], j["repeat"])
    logger.info("model's own next-code accuracy (final layer vs teacher-forcing target): %.3f", out["final_gt_top1"])
    logger.info("saved %s.{json,csv,png}", args.out)


def future_metrics(lens_logits: torch.Tensor, final_logits: torch.Tensor) -> dict[str, float]:
    final_arg = final_logits.argmax(-1)
    log_p_final = torch.log_softmax(final_logits, -1)
    log_p_lens = torch.log_softmax(lens_logits, -1)
    kl = (log_p_final.exp() * (log_p_final - log_p_lens)).sum(-1)
    return {"top1": (lens_logits.argmax(-1) == final_arg).float().mean().item(), "kl": kl.mean().item()}


def plot_future(per_future, offsets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, title) in zip(axes, (("top1", "top-1 agreement with final output at p+Δ"), ("kl", "KL(final at p+Δ || lens at p)"))):
        for l, color in zip((12, 16, 18, 20, 22), ("C0", "C1", "C2", "C3", "C4")):
            if l not in per_future["logit"]:
                continue
            ax.plot(offsets, [per_future["logit"][l][d][metric] for d in offsets], "o--", color=color, ms=4, label=f"L{l} logit lens")
            ax.plot(offsets, [per_future["jlens"][l][d][metric] for d in offsets], "s-", color=color, ms=4, label=f"L{l} J-lens")
        ax.set_xlabel("Δ (positions ahead)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot(per_layer, layers, final_layer, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("top1", "top-1 agreement with final output"), ("kl", "KL(final || lens) [nats]"), ("repeat", "argmax == current code")]
    for ax, (metric, title) in zip(axes, panels):
        for kind, style, label in (("logit", "o--", "logit lens"), ("jlens", "s-", "J-lens")):
            ax.plot(layers, [per_layer[kind][l][metric] for l in layers], style, ms=4, label=label)
        ax.axvline(final_layer, color="gray", lw=0.5)
        ax.set_xlabel("layer")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()

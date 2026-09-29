# Estimator adapted from anthropics/jacobian-lens (jlens/fitting.py).
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting Jacobian lenses on ``LLaMAHF``.

Estimator (same as the official lens): for each output dimension, put a
one-hot cotangent at every target position at once and backprop to the
source layers. The gradient at source position ``p`` is then
``sum_{q in T} ∂h_final[q] / ∂h_l[p]``; ``J_l`` is its mean over the source
positions ``p in S`` and then over examples.

What differs from the language-model version is the choice of ``S`` and
``T`` (see :func:`motion_to_motion` / :func:`text_to_motion`): the text
prefix attends bidirectionally, so text positions must never be targets.

Cost per example: one forward pass on ``dim_batch`` replicas and
``ceil(d_model / dim_batch)`` backward passes. ``dim_batch`` trades memory for
the number of passes; total backward FLOPs do not change.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable, Sequence
from typing import Callable, Tuple

import torch

from jlens_motion.hooks import ActivationRecorder
from jlens_motion.lens import JacobianLens, OffsetJacobians
from jlens_motion.model import (
    MotionExample,
    MotionLensModel,
    motion_positions,
    text_positions,
)

logger = logging.getLogger(__name__)

# typing aliases: this runs under Python 3.8 (docker-shell/Dockerfile).
PositionFn = Callable[[MotionExample], Tuple[torch.Tensor, torch.Tensor]]


def motion_to_motion(*, skip_first: int = 0) -> PositionFn:
    """M-lens: sources and targets are the motion positions."""

    def fn(example: MotionExample) -> tuple[torch.Tensor, torch.Tensor]:
        pos = motion_positions(example, skip_first=skip_first)
        return pos, pos

    return fn


def text_to_motion(*, skip_first: int = 0) -> PositionFn:
    """Sources are the text positions, targets the motion positions: what
    the prompt is poised to make the model generate."""

    def fn(example: MotionExample) -> tuple[torch.Tensor, torch.Tensor]:
        return text_positions(example), motion_positions(example, skip_first=skip_first)

    return fn


POSITION_MODES: dict[str, Callable[..., PositionFn]] = {
    "motion": motion_to_motion,
    "text_to_motion": text_to_motion,
}


def resolve_layers(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve None/negative indices and enforce ``source < target``."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {n_layers} layers")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= target:
        raise ValueError(
            f"source_layers must lie in [0, {target}) (target_layer={target}); got {sorted(source_layers)}"
        )
    return sources, target


def _as_positions(positions: Sequence[int] | torch.Tensor, seq_len: int, name: str) -> torch.Tensor:
    pos = torch.as_tensor(positions, dtype=torch.long).flatten()
    if pos.numel() == 0:
        raise ValueError(f"{name} is empty")
    if pos.min() < 0 or pos.max() >= seq_len:
        raise ValueError(f"{name} out of range for sequence length {seq_len}")
    return pos


def jacobian_for_example(
    model: MotionLensModel,
    example: MotionExample,
    source_layers: Sequence[int] | None,
    *,
    source_positions: Sequence[int] | torch.Tensor,
    target_positions: Sequence[int] | torch.Tensor,
    target_layer: int | None = None,
    dim_batch: int = 8,
) -> dict[int, torch.Tensor]:
    """Per-example estimator of ``J_l`` for each source layer.

    Returns:
        ``{layer: Tensor[d_model, d_model]}`` (fp32, CPU); row ``i`` is the
        gradient of output dimension ``i`` of the target layer.
    """
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    src = _as_positions(source_positions, example.seq_len, "source_positions")
    tgt = _as_positions(target_positions, example.seq_len, "target_positions")
    d_model = model.d_model
    n_passes = math.ceil(d_model / dim_batch)
    jacobians = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in sources}

    idx, clip, y_mask = model.build_inputs(example, batch=dim_batch)
    recorder = ActivationRecorder(model.layers, at=[*sources, target], start_graph_at=min(sources))
    with recorder as rec, torch.enable_grad():
        model.forward(idx, clip, y_mask)
        target_act = rec.activations[target]  # [dim_batch, seq_len, d_model]
        source_acts = [rec.activations[l] for l in sources]
        device = target_act.device
        tgt_d = tgt.to(device)
        rows = torch.arange(dim_batch, device=device)
        cotangent = torch.zeros_like(target_act)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[rows[:n, None], tgt_d[None, :], dim_start + rows[:n, None]] = 1.0
            grads = torch.autograd.grad(
                outputs=target_act,
                inputs=source_acts,
                grad_outputs=cotangent,
                retain_graph=pass_idx < n_passes - 1,
            )
            for layer, grad in zip(sources, grads):
                picked = grad[:n, src.to(grad.device), :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n] = picked.cpu()
            del grads
    return jacobians


def offset_jacobians_for_example(
    model: MotionLensModel,
    example: MotionExample,
    source_layers: Sequence[int] | None,
    *,
    offsets: Sequence[int],
    target_positions: Sequence[int] | torch.Tensor,
    source_positions: Sequence[int] | torch.Tensor | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, int]]:
    """Offset-resolved estimator: ``∂h_final[q] / ∂h_l[q - Δ]`` for each Δ.

    The averaged ``J_l`` mixes every future position, so it cannot say *how
    far ahead* layer ``l`` has committed. Here each target position ``q`` gets
    its own cotangent; one set of backward passes for ``q`` yields the
    gradient at every earlier position at once, i.e. every Δ.

    Cost: ``len(target_positions) * ceil(d_model / dim_batch)`` backward
    passes. Memory: ``len(layers) * len(offsets) * d_model**2 * 4`` bytes on
    the CPU (3B, 4 layers x 8 offsets: ~1.3 GB).

    Returns:
        ``(sums, counts)``: ``sums[l][Δ]`` is the sum over contributing
        ``(q, q-Δ)`` pairs, ``counts[Δ]`` the number of pairs. Pairs whose
        source ``q-Δ`` is not in ``source_positions`` (default: the motion
        positions) are skipped.
    """
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    offsets = sorted(set(int(o) for o in offsets))
    if not offsets or offsets[0] < 0:
        raise ValueError("offsets must be non-empty and >= 0 (the motion stream is causal)")
    tgt = _as_positions(target_positions, example.seq_len, "target_positions")
    if source_positions is None:
        source_positions = motion_positions(example, drop_last=False)
    allowed = set(_as_positions(source_positions, example.seq_len, "source_positions").tolist())

    d_model = model.d_model
    sums = {l: {o: torch.zeros(d_model, d_model) for o in offsets} for l in sources}
    counts = {o: 0 for o in offsets}
    pairs = [(q, o, q - o) for q in tgt.tolist() for o in offsets if (q - o) in allowed]
    if not pairs:
        return sums, counts
    for _, o, _ in pairs:
        counts[o] += 1

    by_target: dict[int, list[tuple[int, int]]] = {}
    for q, o, p in pairs:
        by_target.setdefault(q, []).append((o, p))
    targets = sorted(by_target)
    n_dim_passes = math.ceil(d_model / dim_batch)
    total_passes = len(targets) * n_dim_passes

    idx, clip, y_mask = model.build_inputs(example, batch=dim_batch)
    recorder = ActivationRecorder(model.layers, at=[*sources, target], start_graph_at=min(sources))
    with recorder as rec, torch.enable_grad():
        model.forward(idx, clip, y_mask)
        target_act = rec.activations[target]
        source_acts = [rec.activations[l] for l in sources]
        device = target_act.device
        rows = torch.arange(dim_batch, device=device)
        cotangent = torch.zeros_like(target_act)

        pass_idx = 0
        for q in targets:
            for dim_start in range(0, d_model, dim_batch):
                n = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[rows[:n], q, dim_start + rows[:n]] = 1.0
                pass_idx += 1
                grads = torch.autograd.grad(
                    outputs=target_act,
                    inputs=source_acts,
                    grad_outputs=cotangent,
                    retain_graph=pass_idx < total_passes,
                )
                for layer, grad in zip(sources, grads):
                    for o, p in by_target[q]:
                        sums[layer][o][dim_start : dim_start + n] += grad[:n, p, :].float().cpu()
                del grads
    return sums, counts


def fit(
    model: MotionLensModel,
    examples: Iterable[MotionExample],
    source_layers: Sequence[int] | None = None,
    *,
    positions: PositionFn,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_examples: int | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
    meta: dict | None = None,
) -> JacobianLens:
    """Average :func:`jacobian_for_example` over ``examples``.

    ``positions(example)`` returns ``(source_positions, target_positions)``.
    With ``checkpoint_path`` the running sum is saved every
    ``checkpoint_every`` examples and a rerun resumes after the last saved
    example; this assumes ``examples`` yields the same order on every run.
    """
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    d_model = model.d_model
    jac_sum = {l: torch.zeros(d_model, d_model) for l in sources}
    n_done = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["source_layers"] != sources or state["d_model"] != d_model:
            raise ValueError(f"checkpoint {checkpoint_path} was written with different layers/d_model")
        jac_sum = {int(l): J for l, J in state["jac_sum"].items()}
        n_done = state["n_done"]
        logger.info("resuming from %s after %d examples", checkpoint_path, n_done)

    def save_checkpoint() -> None:
        tmp = f"{checkpoint_path}.tmp"
        torch.save({"jac_sum": jac_sum, "n_done": n_done, "source_layers": sources, "d_model": d_model}, tmp)
        os.replace(tmp, checkpoint_path)

    sqrt_d = math.sqrt(d_model)
    for i, example in enumerate(examples):
        if max_examples is not None and n_done >= max_examples:
            break
        if i < n_done:  # already folded in by the checkpoint
            continue
        start = time.time()
        src, tgt = positions(example)
        per_example = jacobian_for_example(
            model,
            example,
            sources,
            source_positions=src,
            target_positions=tgt,
            target_layer=target,
            dim_batch=dim_batch,
        )
        # Relative change of the running mean: falls roughly as 1/n once settled.
        if n_done > 0:
            rel_change = max(
                ((per_example[l] - jac_sum[l] / n_done).norm() / ((n_done + 1) * (jac_sum[l] / n_done).norm())).item()
                for l in sources
            )
        else:
            rel_change = float("nan")
        for l in sources:
            jac_sum[l] += per_example[l]
        n_done += 1
        logger.info(
            "example %d  len=%d (text %d)  %.1fs  max||J||/sqrt(d)=%.3f  rel_change=%.2e",
            n_done,
            example.seq_len,
            example.text_len,
            time.time() - start,
            max(per_example[l].norm().item() for l in sources) / sqrt_d,
            rel_change,
        )
        if checkpoint_path and n_done % checkpoint_every == 0:
            save_checkpoint()

    if n_done == 0:
        raise ValueError("no examples were fitted")
    if checkpoint_path:
        save_checkpoint()
    info = {"target_layer": target, "dim_batch": dim_batch, **(meta or {})}
    return JacobianLens({l: jac_sum[l] / n_done for l in sources}, n_done, d_model, info)


def fit_offsets(
    model: MotionLensModel,
    examples: Iterable[MotionExample],
    source_layers: Sequence[int],
    *,
    offsets: Sequence[int],
    targets_per_example: int = 4,
    skip_first: int = 0,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_examples: int | None = None,
    seed: int = 0,
    meta: dict | None = None,
) -> OffsetJacobians:
    """Accumulate :func:`offset_jacobians_for_example` over ``examples``.

    For each example, ``targets_per_example`` motion positions are sampled
    uniformly (seeded) among those at least ``max(offsets)`` steps into the
    motion, so that every offset has a valid source.
    """
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    offsets = sorted(set(int(o) for o in offsets))
    generator = torch.Generator().manual_seed(seed)
    sums = {l: {o: torch.zeros(model.d_model, model.d_model) for o in offsets} for l in sources}
    counts = {o: 0 for o in offsets}
    n_done = 0
    for example in examples:
        if max_examples is not None and n_done >= max_examples:
            break
        candidates = motion_positions(example, skip_first=skip_first)
        candidates = candidates[candidates - max(offsets) >= example.text_len + skip_first]
        if candidates.numel() == 0:
            logger.info("skipping example (motion too short for offset %d)", max(offsets))
            continue
        k = min(targets_per_example, candidates.numel())
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:k]]
        start = time.time()
        ex_sums, ex_counts = offset_jacobians_for_example(
            model,
            example,
            sources,
            offsets=offsets,
            target_positions=chosen,
            target_layer=target,
            dim_batch=dim_batch,
        )
        for l in sources:
            for o in offsets:
                sums[l][o] += ex_sums[l][o]
        for o in offsets:
            counts[o] += ex_counts[o]
        n_done += 1
        logger.info("example %d  %d targets  %.1fs", n_done, k, time.time() - start)
    if n_done == 0:
        raise ValueError("no examples were fitted")
    info = {"target_layer": target, "n_examples": n_done, **(meta or {})}
    return OffsetJacobians(sums, counts, model.d_model, info)

# Parts adapted from anthropics/jacobian-lens (jlens/lens.py).
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitted Jacobian lenses and the readout ``lens_l(h) = unembed(J_l @ h)``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from jlens_motion.hooks import ActivationRecorder
from jlens_motion.model import MotionExample, MotionLensModel


class JacobianLens:
    """Per-layer ``J_l`` matrices (``[d_model, d_model]``, fp32).

    Attributes:
        jacobians: ``{source_layer: J_l}``.
        n_examples: Number of examples averaged into each ``J_l``.
        d_model: Residual width.
        meta: Free-form fit settings (position mode, target layer, ...),
            saved alongside the matrices.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        n_examples: int,
        d_model: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        for layer, J in jacobians.items():
            if tuple(J.shape) != (d_model, d_model):
                raise ValueError(f"J_{layer} has shape {tuple(J.shape)}, expected {(d_model, d_model)}")
        self.jacobians = jacobians
        self.n_examples = n_examples
        self.d_model = d_model
        self.meta = dict(meta or {})

    def __repr__(self) -> str:
        return (
            f"JacobianLens(layers={self.source_layers}, d_model={self.d_model}, "
            f"n_examples={self.n_examples})"
        )

    @property
    def source_layers(self) -> list[int]:
        return sorted(self.jacobians)

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "jacobians": {l: J.to(dtype) for l, J in self.jacobians.items()},
                "n_examples": self.n_examples,
                "d_model": self.d_model,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            jacobians={int(l): J.float() for l, J in state["jacobians"].items()},
            n_examples=state["n_examples"],
            d_model=state["d_model"],
            meta=state.get("meta"),
        )

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """``n_examples``-weighted mean of lenses fitted on disjoint shards."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_examples for lens in lenses)
        merged = {
            layer: sum(lens.jacobians[layer] * lens.n_examples for lens in lenses) / n_total
            for layer in first.source_layers
        }
        return cls(merged, n_total, first.d_model, first.meta)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l @ h`` for a residual of shape ``[..., d_model]``."""
        J = self.jacobians[layer].to(residual.device)
        return residual.float() @ J.T


class OffsetJacobians:
    """Offset-resolved Jacobians ``J_l^(Δ) = E[∂h_final,p+Δ / ∂h_l,p]``.

    Stored as running sums plus per-offset pair counts so that shards and
    examples can be merged exactly; :meth:`mean` gives the averages.
    """

    def __init__(
        self,
        sums: dict[int, dict[int, torch.Tensor]],
        counts: dict[int, int],
        d_model: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.sums = sums
        self.counts = counts
        self.d_model = d_model
        self.meta = dict(meta or {})

    @property
    def source_layers(self) -> list[int]:
        return sorted(self.sums)

    @property
    def offsets(self) -> list[int]:
        return sorted(self.counts)

    def mean(self, layer: int, offset: int) -> torch.Tensor:
        count = self.counts[offset]
        if count == 0:
            raise ValueError(f"no (source, target) pairs were seen at offset {offset}")
        return self.sums[layer][offset] / count

    def transport(self, residual: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        J = self.mean(layer, offset).to(residual.device)
        return residual.float() @ J.T

    def save(self, path: str) -> None:
        torch.save(
            {"sums": self.sums, "counts": self.counts, "d_model": self.d_model, "meta": self.meta},
            path,
        )

    @classmethod
    def load(cls, path: str) -> OffsetJacobians:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state["sums"], state["counts"], state["d_model"], state.get("meta"))


@torch.no_grad()
def lens_readout(
    model: MotionLensModel,
    example: MotionExample,
    layers: Sequence[int],
    *,
    lens: JacobianLens | None = None,
    positions: Sequence[int] | torch.Tensor | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Read out every requested layer at ``positions``.

    With ``lens=None`` this is the plain logit lens (``unembed(h_l)``), the
    baseline the Jacobian lens must beat.

    Returns:
        ``(lens_logits, model_logits)``: ``lens_logits[l]`` and
        ``model_logits`` are ``[n_positions, nb_code + 1]`` fp32 CPU tensors;
        ``model_logits`` is the model's own output at the same positions.
    """
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    idx, clip, y_mask = model.build_inputs(example)
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(idx, clip, y_mask)
        acts = {l: recorder.activations[l][0].detach() for l in record_at}

    pos = None if positions is None else torch.as_tensor(positions, dtype=torch.long)

    def select(layer: int) -> torch.Tensor:
        full = acts[layer]
        return full if pos is None else full[pos.to(full.device)]

    lens_logits = {}
    for layer in layers:
        residual = select(layer)
        # J at the target (final) layer is the identity by definition.
        if lens is not None and layer != final_layer:
            residual = lens.transport(residual, layer)
        lens_logits[layer] = model.unembed(residual).float().cpu()
    model_logits = model.unembed(select(final_layer)).float().cpu()
    return lens_logits, model_logits

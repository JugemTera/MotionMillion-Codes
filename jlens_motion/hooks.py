# Adapted from anthropics/jacobian-lens (jlens/hooks.py).
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward hooks that capture the residual stream after each block."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


class ActivationRecorder:
    """Capture the output of ``blocks[i]`` for each ``i`` in ``at``.

    ``Block.forward`` in ``models/lit_llama/model_hf.py`` returns the residual
    stream as a plain tensor, so the hooked output *is* ``h_l``.

    Captured tensors are not detached, so they can be passed straight to
    :func:`torch.autograd.grad`. If ``start_graph_at`` is given, the tensor
    captured there is marked ``requires_grad_(True)``; with every model
    parameter frozen this makes it the root of the autograd graph, so the
    retained graph only spans the blocks after it.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

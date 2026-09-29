"""Adapter that exposes ``LLaMAHF`` (models/lit_llama/model_hf.py) to the lens.

Sequence layout for one example (batch size 1)::

    position:  0 ........ Lt-1 | Lt ...... Lt+N-1
    input:     T5 text features | c_0 ...... c_{N-1}
    attention: bidirectional    | causal

The residual at position ``p`` predicts the token at ``p + 1``: the last text
position predicts ``c_0`` and motion position ``Lt + k`` (which has read
``c_k``) predicts ``c_{k+1}``. No end token is appended.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MotionExample:
    """One text/motion pair.

    Attributes:
        text_feat: T5 last hidden state for the valid (unpadded) text tokens,
            shape ``[Lt, clip_dim]``.
        codes: Motion token ids in ``[0, nb_code)``, shape ``[N]``.
        caption: The raw caption, kept for logging only.
    """

    text_feat: torch.Tensor
    codes: torch.Tensor
    caption: str = ""

    @property
    def text_len(self) -> int:
        return int(self.text_feat.shape[0])

    @property
    def n_motion(self) -> int:
        return int(self.codes.shape[0])

    @property
    def seq_len(self) -> int:
        return self.text_len + self.n_motion


def text_positions(example: MotionExample) -> torch.Tensor:
    """Positions holding the injected T5 features: ``0 .. Lt-1``."""
    return torch.arange(example.text_len)


def motion_positions(
    example: MotionExample, *, skip_first: int = 0, drop_last: bool = True
) -> torch.Tensor:
    """Positions holding motion codes: ``Lt+skip_first .. Lt+N-1``.

    ``drop_last`` removes the final position, whose next-token target lies
    outside the given codes (the official lens drops it for the same reason).
    """
    start = example.text_len + skip_first
    stop = example.seq_len - (1 if drop_last else 0)
    if stop <= start:
        raise ValueError(
            f"no motion positions left: text_len={example.text_len}, "
            f"n_motion={example.n_motion}, skip_first={skip_first}, drop_last={drop_last}"
        )
    return torch.arange(start, stop)


class MotionLensModel:
    """Wrap a loaded ``LLaMAHF`` for Jacobian-lens fitting and readout.

    The constructor puts the model in eval mode and freezes every parameter
    in place (the fit needs gradients only with respect to activations).
    """

    def __init__(self, net: nn.Module) -> None:
        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)
        self.net = net
        self.config = net.config
        self.layers: nn.ModuleList = net.transformer.h
        self.n_layers: int = self.config.n_layer
        self.d_model: int = self.config.n_embd
        # vocab_size = nb_code + 2; lm_head predicts nb_code codes + end token.
        self.nb_code: int = self.config.vocab_size - 2
        self.end_idx: int = self.nb_code
        self.pad_idx: int = self.nb_code + 1
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.n_layer={self.n_layers} but found {len(self.layers)} blocks"
            )

    def __repr__(self) -> str:
        return (
            f"MotionLensModel(n_layers={self.n_layers}, d_model={self.d_model}, "
            f"nb_code={self.nb_code})"
        )

    @property
    def device(self) -> torch.device:
        return self.net.transformer.wte.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.net.transformer.wte.weight.dtype

    def build_inputs(
        self, example: MotionExample, batch: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(idx, clip_feature, y_mask)`` as ``LLaMAHF.forward`` expects.

        ``idx`` carries pad tokens at the text positions; they are overwritten
        by ``llama_proj(clip_feature)`` inside the forward pass, exactly as in
        training (``dataset/dataset_TM_train_motionmillion.py``). The single
        example is replicated ``batch`` times along the batch axis.
        """
        codes = example.codes.to(self.device, torch.long)
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.nb_code):
            raise ValueError(f"motion codes must lie in [0, {self.nb_code})")
        text_len = example.text_len
        pad = torch.full((text_len,), self.pad_idx, dtype=torch.long, device=self.device)
        idx = torch.cat([pad, codes])[None].expand(batch, -1)
        clip = example.text_feat.to(self.device, self.dtype)[None].expand(batch, -1, -1)
        y_mask = torch.ones(batch, text_len, dtype=torch.long, device=self.device)
        if idx.shape[1] > self.config.block_size:
            raise ValueError(
                f"sequence length {idx.shape[1]} exceeds block_size {self.config.block_size}"
            )
        return idx, clip, y_mask

    def forward(
        self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the residual stack and return the last block's output (no final
        norm, no LM head).

        Mirrors ``LLaMAHF.forward`` up to the block loop; skipping ``lm_head``
        avoids materialising 65k-way logits for every replicated batch row.
        ``tests/test_jlens_motion.py`` checks that ``unembed(forward(...))``
        equals ``LLaMAHF.forward``.
        """
        net = self.net
        text_length = clip_feature.shape[1]
        x = net.transformer.wte(idx)
        expanded_mask = y_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        text = torch.where(
            expanded_mask == 1, net.llama_proj(clip_feature), x[:, :text_length, :]
        )
        x = torch.cat((text, x[:, text_length:, :]), dim=1)
        for block in self.layers:
            x = block(x, y_mask)
        return x

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + LM head: ``[..., d_model] -> [..., nb_code + 1]``."""
        net = self.net
        weight = net.lm_head.weight
        return net.lm_head(net.transformer.ln_f(residual.to(weight.device, weight.dtype)))

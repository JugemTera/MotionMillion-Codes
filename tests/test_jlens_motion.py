"""Checks for jlens_motion on a tiny random LLaMAHF (CPU, no checkpoints).

Run from the repository root:  python -m pytest tests/test_jlens_motion.py
"""

import pytest
import torch

from jlens_motion import (
    ActivationRecorder,
    JacobianLens,
    MotionExample,
    MotionLensModel,
    fit,
    jacobian_for_example,
    lens_readout,
    motion_positions,
    motion_to_motion,
    offset_jacobians_for_example,
    text_positions,
    text_to_motion,
)
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig

NB_CODE = 30
CLIP_DIM = 8
D_MODEL = 16
N_LAYER = 4


def make_model(seed: int = 0) -> MotionLensModel:
    torch.manual_seed(seed)
    config = LLaMAHFConfig(
        block_size=64, vocab_size=NB_CODE + 2, n_layer=N_LAYER, n_head=2, n_embd=D_MODEL
    )
    config.clip_dim = CLIP_DIM
    config.tie_weights = False
    net = LLaMAHF(config)
    # The default init is tiny (std 0.02); scale weights up so the blocks are
    # visibly non-linear and the checks below are not trivially satisfied.
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() > 1:
                p.mul_(10.0)
    return MotionLensModel(net)


def make_example(text_len: int = 4, n_motion: int = 7, seed: int = 1) -> MotionExample:
    g = torch.Generator().manual_seed(seed)
    return MotionExample(
        text_feat=torch.randn(text_len, CLIP_DIM, generator=g),
        codes=torch.randint(0, NB_CODE, (n_motion,), generator=g),
    )


def bruteforce_jacobian(model, example, layer, src, tgt):
    """mean_{p in src} sum_{q in tgt} ∂h_final[q] / ∂h_layer[p], computed by
    re-running the downstream blocks under torch.autograd.functional.jacobian."""
    idx, clip, y_mask = model.build_inputs(example)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(idx, clip, y_mask)
        h = rec.activations[layer][0].clone()

    def downstream(h_l):
        x = h_l[None]
        for block in model.layers[layer + 1 :]:
            x = block(x, y_mask)
        return x[0]

    jac = torch.autograd.functional.jacobian(downstream, h)  # [T, d, T, d]
    src, tgt = torch.as_tensor(src), torch.as_tensor(tgt)
    return jac[tgt][:, :, src].sum(dim=0).mean(dim=1)  # [d, d]


def test_forward_matches_llamahf():
    model = make_model()
    ex = make_example()
    idx, clip, y_mask = model.build_inputs(ex)
    with torch.no_grad():
        ours = model.unembed(model.forward(idx, clip, y_mask))
        reference = model.net(idx, clip, y_mask)
    torch.testing.assert_close(ours, reference)


@pytest.mark.parametrize("mode", ["motion", "text_to_motion"])
def test_jacobian_matches_bruteforce(mode):
    model = make_model()
    ex = make_example()
    src, tgt = (motion_to_motion() if mode == "motion" else text_to_motion())(ex)
    layers = [0, 2]
    # dim_batch=3 does not divide d_model=16, so the ragged last pass is covered.
    ours = jacobian_for_example(
        model, ex, layers, source_positions=src, target_positions=tgt, dim_batch=3
    )
    for layer in layers:
        ref = bruteforce_jacobian(model, ex, layer, src, tgt)
        assert ref.abs().max() > 1e-3
        torch.testing.assert_close(ours[layer], ref, rtol=1e-4, atol=1e-5)


def test_dim_batch_does_not_change_result():
    model = make_model()
    ex = make_example()
    src, tgt = motion_to_motion()(ex)
    a = jacobian_for_example(model, ex, [1], source_positions=src, target_positions=tgt, dim_batch=1)
    b = jacobian_for_example(model, ex, [1], source_positions=src, target_positions=tgt, dim_batch=5)
    torch.testing.assert_close(a[1], b[1], rtol=1e-4, atol=1e-6)


def test_motion_never_reaches_text():
    """Text positions attend only to text, so motion residuals cannot move
    them: the Jacobian from motion sources to text targets is exactly zero."""
    model = make_model()
    ex = make_example()
    J = jacobian_for_example(
        model,
        ex,
        [0],
        source_positions=motion_positions(ex, drop_last=False),
        target_positions=text_positions(ex),
    )
    assert J[0].abs().max() == 0


def test_offset_jacobians_match_bruteforce():
    model = make_model()
    ex = make_example(n_motion=9)
    offsets = [0, 1, 3]
    targets = [ex.text_len + 5, ex.text_len + 7]
    sums, counts = offset_jacobians_for_example(
        model, ex, [1], offsets=offsets, target_positions=targets, dim_batch=4
    )
    for o in offsets:
        assert counts[o] == len(targets)
        ref = sum(bruteforce_jacobian(model, ex, 1, [q - o], [q]) for q in targets)
        torch.testing.assert_close(sums[1][o], ref, rtol=1e-4, atol=1e-5)


def test_readout_identity_lens_equals_logit_lens():
    model = make_model()
    ex = make_example()
    eye = JacobianLens({l: torch.eye(D_MODEL) for l in range(N_LAYER - 1)}, 1, D_MODEL)
    layers = list(range(N_LAYER))
    plain, model_logits = lens_readout(model, ex, layers)
    with_eye, _ = lens_readout(model, ex, layers, lens=eye)
    for l in layers:
        torch.testing.assert_close(plain[l], with_eye[l])
    # The final-layer readout is the model's own output.
    torch.testing.assert_close(plain[N_LAYER - 1], model_logits)
    idx, clip, y_mask = model.build_inputs(ex)
    with torch.no_grad():
        torch.testing.assert_close(model_logits, model.net(idx, clip, y_mask)[0])


def test_fit_resume_merge_and_save(tmp_path):
    model = make_model()
    examples = [make_example(seed=s) for s in range(4)]
    kwargs = dict(positions=motion_to_motion(), dim_batch=8)

    full = fit(model, examples, [0, 1], **kwargs)
    per_example = [
        jacobian_for_example(model, ex, [0, 1], source_positions=s, target_positions=t)
        for ex in examples
        for s, t in [motion_to_motion()(ex)]
    ]
    for l in [0, 1]:
        torch.testing.assert_close(full.jacobians[l], sum(j[l] for j in per_example) / 4)

    # Interrupted after 2 examples, then resumed from the checkpoint.
    ckpt = str(tmp_path / "ckpt.pt")
    fit(model, examples, [0, 1], max_examples=2, checkpoint_path=ckpt, checkpoint_every=1, **kwargs)
    resumed = fit(model, examples, [0, 1], checkpoint_path=ckpt, **kwargs)
    assert resumed.n_examples == 4
    for l in [0, 1]:
        torch.testing.assert_close(resumed.jacobians[l], full.jacobians[l])

    # Shards merged by example count equal the full fit.
    merged = JacobianLens.merge(
        [fit(model, examples[:1], [0, 1], **kwargs), fit(model, examples[1:], [0, 1], **kwargs)]
    )
    for l in [0, 1]:
        torch.testing.assert_close(merged.jacobians[l], full.jacobians[l])

    path = str(tmp_path / "lens.pt")
    full.save(path, dtype=torch.float32)
    loaded = JacobianLens.load(path)
    assert loaded.source_layers == [0, 1] and loaded.n_examples == 4
    torch.testing.assert_close(loaded.jacobians[1], full.jacobians[1])

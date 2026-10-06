"""Fit a Jacobian lens (J-lens) on the MotionMillion text-to-motion LLaMA.

Examples are (caption, motion-code sequence) pairs sampled from
``{data_root}/all_data.pkl`` (the same file the training loader reads). The
captions are encoded with flan-t5-xl first, then T5 is freed before fitting.

    python fit_jlens.py --pretrained_llama 3B \
        --resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth \
        --mode motion --n-examples 100 --out results/jlens/3B_motion.pt

Modes:
    motion          sources = targets = motion positions (M-lens)
    text_to_motion  sources = text positions, targets = motion positions
    offset          offset-resolved J_l^(Δ) at --layers for --offsets

See jlens_motion/README.md for the definitions and cost.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random

import torch

from jlens_motion import (
    POSITION_MODES,
    MotionExample,
    MotionLensModel,
    fit,
    fit_offsets,
)
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig

logger = logging.getLogger("fit_jlens")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # model (names follow options/option_transformer.py)
    parser.add_argument("--pretrained_llama", default="3B")
    parser.add_argument("--resume-trans", required=True, help="LLaMA checkpoint (.pth with a 'trans' entry)")
    parser.add_argument("--nb-code", type=int, default=64000, help="FSQ codebook size: prod([8,8,8,5,5,5]) = 64000 (the '65536' in the training scripts only selects that levels branch)")
    parser.add_argument("--block-size", type=int, default=301)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--t5-path", default="checkpoints/flan-t5-xl")
    parser.add_argument("--clip-dim", type=int, default=2048, help="2048 for flan-t5-xl, 4096 for xxl")
    parser.add_argument("--dtype", choices=DTYPES, default="fp32", help="model dtype (bf16 halves memory but coarsens the gradients); J is accumulated in fp32")
    parser.add_argument("--device", default="cuda:0")
    # data
    parser.add_argument("--data-root", default="./dataset/MotionMillion")
    parser.add_argument("--split-file", default="./dataset/MotionMillion/split/version1/t2m_60_300/val.txt")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--max-text-length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    # lens
    parser.add_argument("--mode", choices=[*POSITION_MODES, "offset"], default="motion")
    parser.add_argument("--layers", default="all", help="'all' (every layer below the target) or e.g. '6,12,18'")
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--skip-first", type=int, default=0, help="leading motion positions to drop")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--offsets", default="0,1,2,4,8,16", help="offset mode only")
    parser.add_argument("--targets-per-example", type=int, default=4, help="offset mode only")
    # output
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None, help="resumable running sum (motion/text_to_motion)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def load_model(args) -> MotionLensModel:
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size
    config.vocab_size = args.nb_code + 2
    config.clip_dim = args.clip_dim
    config.tie_weights = args.tie_weights
    net = LLaMAHF(config)
    ckpt = torch.load(args.resume_trans, map_location="cpu")
    ckpt = {k.replace("module.", ""): v for k, v in ckpt["trans"].items()}
    net.load_state_dict(ckpt, strict=True)
    net.to(device=args.device, dtype=DTYPES[args.dtype])
    return MotionLensModel(net)


def sample_pairs(args) -> list[tuple[str, str, list[int]]]:
    """Deterministically sample (name, caption, codes) from all_data.pkl."""
    with open(os.path.join(args.data_root, "all_data.pkl"), "rb") as f:
        all_data = pickle.load(f)
    with open(args.split_file) as f:
        names = [line.strip() for line in f if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(names)
    pairs = []
    for name in names:
        captions = [c.strip() for c in all_data["text_data"].get(name, []) if c.strip()]
        code_lists = all_data["code_data"].get(name, [])
        if not captions or not len(code_lists):
            continue
        codes = rng.choice(code_lists)
        pairs.append((name, rng.choice(captions), [int(c) for c in codes]))
        if len(pairs) == args.n_examples:
            break
    if not pairs:
        raise SystemExit(
            f"no names in {args.split_file} have both captions and codes in all_data.pkl "
            "(all_data.pkl is built by train_t2m_get_codes.py from its own split file)"
        )
    if len(pairs) < args.n_examples:
        logger.warning("only %d of %d requested examples are available", len(pairs), args.n_examples)
    return pairs


@torch.no_grad()
def encode_examples(args, pairs) -> list[MotionExample]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(args.t5_path, local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(args.t5_path, local_files_only=True).to(args.device).eval()
    examples = []
    for name, caption, codes in pairs:
        inputs = tokenizer(caption, truncation=True, return_tensors="pt")
        feat = encoder(input_ids=inputs.input_ids.to(args.device)).last_hidden_state[0]
        feat = feat[: args.max_text_length].float().cpu()
        # Keep the whole sequence inside the model's block size.
        codes = codes[: args.block_size - feat.shape[0]]
        examples.append(MotionExample(feat, torch.tensor(codes, dtype=torch.long), caption))
        logger.info("%s  text=%d motion=%d  %s", name, feat.shape[0], len(codes), caption)
    del encoder
    torch.cuda.empty_cache()
    return examples


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    pairs = sample_pairs(args)
    logger.info("sampled %d examples from %s", len(pairs), args.split_file)
    examples = encode_examples(args, pairs)
    model = load_model(args)
    logger.info("%s, dtype=%s", model, args.dtype)

    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    meta = {
        "mode": args.mode,
        "pretrained_llama": args.pretrained_llama,
        "resume_trans": args.resume_trans,
        "split_file": args.split_file,
        "seed": args.seed,
        "skip_first": args.skip_first,
        "captions": [e.caption for e in examples],
    }

    if args.mode == "offset":
        if layers is None:
            raise SystemExit("--mode offset needs an explicit --layers list (memory grows with layers x offsets)")
        offsets = [int(x) for x in args.offsets.split(",")]
        result = fit_offsets(
            model,
            examples,
            layers,
            offsets=offsets,
            targets_per_example=args.targets_per_example,
            skip_first=args.skip_first,
            target_layer=args.target_layer,
            dim_batch=args.dim_batch,
            seed=args.seed,
            meta=meta,
        )
        result.save(args.out)
    else:
        lens = fit(
            model,
            examples,
            layers,
            positions=POSITION_MODES[args.mode](skip_first=args.skip_first),
            target_layer=args.target_layer,
            dim_batch=args.dim_batch,
            checkpoint_path=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
            meta=meta,
        )
        lens.save(args.out)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()

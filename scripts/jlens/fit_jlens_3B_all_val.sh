# NOTE: the FSQ codebook is 8*8*8*5*5*5 = 64,000 codes (vocab 64,002 with END/PAD).
#       "65536" in the training scripts only selects the levels branch in vqvae.py.
# M-lens (motion -> motion) on every layer of the 3B all.txt model, examples from
# the available val split. N_EXAMPLES=3 first to measure the per-example time.
#   N_EXAMPLES=3   bash scripts/jlens/fit_jlens_3B_all_val.sh
#   N_EXAMPLES=1000 bash scripts/jlens/fit_jlens_3B_all_val.sh
# Resumable: the running sum is checkpointed every 10 examples to --checkpoint,
# so rerunning the same command continues after the last saved example.
N_EXAMPLES=${N_EXAMPLES:-1000}
MODE=${MODE:-motion}
OUT_DIR=results/jlens
mkdir -p $OUT_DIR
python fit_jlens.py \
--pretrained_llama 3B \
--resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth \
--nb-code 64000 \
--block-size 301 \
--t5-path checkpoints/flan-t5-xl \
--clip-dim 2048 \
--dtype fp32 \
--data-root ./dataset/MotionMillion \
--split-file ./dataset/MotionMillion/split/version1_avail/t2m_60_300/val.txt \
--n-examples ${N_EXAMPLES} \
--seed 0 \
--mode ${MODE} \
--dim-batch 8 \
--out ${OUT_DIR}/3B_all_val_${MODE}_n${N_EXAMPLES}.pt \
--checkpoint ${OUT_DIR}/3B_all_val_${MODE}_n${N_EXAMPLES}.ckpt \
--checkpoint-every 10

# M-lens (motion -> motion) on every layer of the 3B model.
# Measure the per-example time on a few examples first (e.g. --n-examples 3).
python fit_jlens.py \
--pretrained_llama 3B \
--resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth \
--nb-code 64000 \
--block-size 301 \
--t5-path checkpoints/flan-t5-xl \
--clip-dim 2048 \
--dtype fp32 \
--data-root ./dataset/MotionMillion \
--split-file ./dataset/MotionMillion/split/version1/t2m_60_300/val.txt \
--n-examples 100 \
--seed 0 \
--mode motion \
--dim-batch 8 \
--out results/jlens/3B_motion.pt \
--checkpoint results/jlens/3B_motion.ckpt

# Offset-resolved Jacobians at a few layers:
# python fit_jlens.py ... --mode offset --layers 6,12,18 --offsets 0,1,2,4,8,16 \
#   --targets-per-example 4 --out results/jlens/3B_offset.pt

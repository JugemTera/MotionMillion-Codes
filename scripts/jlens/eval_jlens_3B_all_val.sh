# Logit lens vs J-lens on held-out val examples (docs/07 checks 1 and 2).
#   N_EXAMPLES=3   bash scripts/jlens/eval_jlens_3B_all_val.sh   # smoke test
#   N_EXAMPLES=300 bash scripts/jlens/eval_jlens_3B_all_val.sh
N_EXAMPLES=${N_EXAMPLES:-300}
MODE=${MODE:-motion}
LENS=${LENS:-results/jlens/3B_all_val_${MODE}_n1000.pt}
OUT_DIR=results/jlens
mkdir -p $OUT_DIR
python eval_jlens.py \
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
--seed 1 \
--exclude-fit-seed 0 --exclude-fit-n 1000 \
--lens ${LENS} \
--out ${OUT_DIR}/eval_3B_all_val_${MODE}_n1000_heldout${N_EXAMPLES}

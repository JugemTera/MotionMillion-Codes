# Batch text-to-motion inference with your own prompts (one prompt per line).
#   PROMPT_FILE=assets/my_prompts.txt EXP_NAME=custom_3B bash scripts/inference/batch_inference/test_t2m_3B_custom.sh
# Results: results/output/inference/batch_inference/${EXP_NAME}/<n>/{0,1}_*.gif
#   0_* = original prompt, 1_* = prompt rewritten by the Llama rewrite model.
# Set USE_REWRITE=0 to skip the rewrite model (only 0_* is produced).
PROMPT_FILE=${PROMPT_FILE:-assets/my_prompts.txt}
EXP_NAME=${EXP_NAME:-custom_3B}
USE_REWRITE=${USE_REWRITE:-1}
if [[ "$USE_REWRITE" == 1 ]]; then
  REWRITE_ARGS="--use_rewrite_model --rewrite_model_path ./checkpoints/rewrite_models/Meta-Llama-3.1-8B-Instruct"
else
  REWRITE_ARGS=""
fi
# strip blank lines: an empty prompt would be fed to the model as-is
grep -v '^[[:space:]]*$' "$PROMPT_FILE" > /tmp/prompts_$$.txt
PROMPT_FILE=/tmp/prompts_$$.txt
echo "prompts: $(wc -l < $PROMPT_FILE)  exp-name: $EXP_NAME  rewrite: $USE_REWRITE"

python inference_batch.py \
--exp-name ${EXP_NAME} \
--batch-size 32 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 65536 \
--n-head-gpt 16 \
--block-size 301 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth ./checkpoints/pretrained_models/fsq_net_6000000.pth \
--vq-name VQVAE_codebook_65536_FSQ_all \
--out-dir results/output/inference/batch_inference/ \
--total-iter 120000 \
--lr-scheduler-type CosineDecayScheduler \
--lr 0.0002 \
--dataname motionmillion \
--down-t 1 \
--depth 3 \
--quantizer FSQ \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-norm LN \
--fps 30 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--text_encode flan-t5-xl \
--pretrained_llama 3B \
--pkeep 1 \
--motion_type vector_272 \
--text_type texts \
--version version1/t2m_60_300 \
--mixed_precision bf16 \
${REWRITE_ARGS} \
--infer_batch_prompt ${PROMPT_FILE} \
--resume-trans ./checkpoints/pretrained_models/motionmillion_3B_all.pth

export NCCL_TIMEOUT=1200

# Number of GPUs to use. Override from the job script, e.g. NUM_GPUS=4.
NUM_GPUS=${NUM_GPUS:-8}
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

# --batch-size is per process, so the global batch is 64 * NUM_GPUS (512 on 8 GPUs).
CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --multi_gpu --num_processes $NUM_GPUS --main_process_port 29701 train_tokenizer.py \
--batch-size 64 \
--lr 5e-5 \
--total-iter 6000000 \
--lr-scheduler 300000 \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir results/output/FSQ_96len_romo \
--dataname romo \
--vq-act relu \
--quantizer FSQ \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name train_VQVAE_FSQ_bz64_lr5e-5_codebook65536_romo_${NUM_GPUS}gpu_96window-size_1down-t_3depth_3kernelsize_48000warmup_wavelet_1patch_layernorm \
--nb-code 65536 \
--motion_type vector_272 \
--version version1/tokenizer_96 \
--eval-version version1/tokenizer_96_eval512 \
--warm-up-iter 48000 \
--num-workers 16 \
--window-size 96 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--vq-norm LN

# Resume: point at the checkpoint written by the previous run (same exp-name dir).
# --resume-pth results/output/FSQ_96len_romo/<exp-name>/net_latest.pth

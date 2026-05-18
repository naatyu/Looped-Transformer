#!/bin/bash

# One-GPU speedrun variant tuned for a 16 GB card.
# Probed locally:
# - d16 was safe but left too much headroom
# - d20 at seq_len=1024 and device_batch_size=8 reached ~14 GB and still ran
# - d24 failed on this card
#
# This keeps the same end-to-end flow as runs/speedrun.sh, but with a smaller
# model and shorter context so it fits a single consumer GPU.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# Hardware / training knobs. Override from the environment if needed.
UV_EXTRA="${UV_EXTRA:-gpu-cu132}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEPTH="${DEPTH:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16384}"
PRETRAIN_ITERS="${PRETRAIN_ITERS:-300}"
SFT_ITERS="${SFT_ITERS:-100}"
EVAL_EVERY="${EVAL_EVERY:-999999}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-999999}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:--1}"
MODEL_TAG="${MODEL_TAG:-d20_16gb}"

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra "$UV_EXTRA"
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Training report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer / data
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$PRETRAIN_ITERS" \
    --eval-every="$EVAL_EVERY" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --save-every="$SAVE_EVERY" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# SFT
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --num-iterations="$SFT_ITERS" \
    --eval-every="$CORE_METRIC_EVERY" \
    --chatcore-every="$CORE_METRIC_EVERY" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft \
    --model-tag="$MODEL_TAG"

# -----------------------------------------------------------------------------
python -m nanochat.report generate

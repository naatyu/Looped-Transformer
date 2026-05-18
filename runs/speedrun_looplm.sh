#!/bin/bash

# LoopLM variant of the standard speedrun.
# Set UV_EXTRA=gpu-cu132 on CUDA 13.2 systems, or leave it unset to use the default CUDA 12.8 path.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra "${UV_EXTRA:-gpu}"
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# LoopLM d24 run: same speedrun flow, but with a looped transformer trunk.
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN \
    --model-impl=looped \
    --num-loops=2 \
    --model-tag="loopd24x2"
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --model-tag="loopd24x2" \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN \
    --model-tag="loopd24x2"
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
    -i sft \
    --model-tag="loopd24x2"

# -----------------------------------------------------------------------------
python -m nanochat.report generate

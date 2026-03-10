#!/bin/bash
# =============================================================================
# ManiFlow RL Training with Robo-Dopamine Dense Reward
#
# This script launches PPO RL training where the reward signal is:
#   final_reward = sparse_reward_bonus × env_discrete_reward
#                + dense_reward_scale × robodopamine_dense_reward
#
# The dense reward model runs as a subprocess in the robo-dopamine conda env
# (torch 2.8 + vLLM), while the RL training runs in the equidiff env (torch 2.1).
#
# GPU memory management: policy is offloaded to CPU during reward computation,
# then restored to GPU for PPO updates. Both fit in 40GB A100.
#
# Usage:
#   conda activate equidiff
#   bash scripts/run_rl_with_dopamine_reward.sh
#
# Prerequisites:
#   - equidiff conda env for RL training
#   - robo-dopamine conda env for reward model (called via subprocess)
#   - Finetuned merged model at MERGED_MODEL_PATH
#     (run scripts/prepare_and_finetune_dopamine.sh first)
# =============================================================================
set -e

# =====================================================
# Configuration — EDIT THESE FOR YOUR TASK
# =====================================================
TASK_NAME="nut_assembly_d0"
TASK_DESCRIPTION="assemble the round nut onto the correct peg"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MERGED_MODEL_PATH="$PROJECT_DIR/checkpoints/${TASK_NAME}_grm_merged"

# Reward model parameters
SPARSE_REWARD_BONUS=10.0    # Weight for environment discrete reward (0/1)
DENSE_REWARD_SCALE=1.0      # Weight for dense VLM reward
GPU_MEMORY_UTIL=0.7         # vLLM GPU memory fraction (leave room for KV cache)
INFERENCE_BATCH_SIZE=2      # Batch size for VLM inference (conservative for 40GB)
PERSPECTIVES="incremental,forward,backward"  # Reward perspectives
CONDA_ENV="robo-dopamine"   # Conda env for reward model subprocess

# RL training parameters
NUM_ENVS=16
TOTAL_TIMESTEPS=2000000
TRAIN_EPISODES=500
EVAL_EPISODES=50

export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo " ManiFlow RL + Dopamine Dense Reward"
echo "=============================================="
echo "  Task:              $TASK_NAME"
echo "  Merged model:      $MERGED_MODEL_PATH"
echo "  Sparse bonus:      $SPARSE_REWARD_BONUS"
echo "  Dense scale:       $DENSE_REWARD_SCALE"
echo "  Perspectives:      $PERSPECTIVES"
echo "  Num envs:          $NUM_ENVS"
echo "  Total timesteps:   $TOTAL_TIMESTEPS"
echo "=============================================="

# Verify merged model exists
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "ERROR: Merged model not found at $MERGED_MODEL_PATH"
    echo "Run scripts/prepare_and_finetune_dopamine.sh first."
    exit 1
fi

# Verify robo-dopamine conda env exists
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "ERROR: Conda env '$CONDA_ENV' not found."
    echo "Create it with the Robo-Dopamine requirements."
    exit 1
fi

cd "$PROJECT_DIR"

# =====================================================
# Launch RL training with dopamine reward enabled
# =====================================================
python train_maniflow_rl.py \
    task_name=$TASK_NAME \
    dopamine_reward.enabled=true \
    dopamine_reward.model_path="$MERGED_MODEL_PATH" \
    dopamine_reward.task_description="$TASK_DESCRIPTION" \
    dopamine_reward.sparse_reward_bonus=$SPARSE_REWARD_BONUS \
    dopamine_reward.dense_reward_scale=$DENSE_REWARD_SCALE \
    dopamine_reward.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    dopamine_reward.inference_batch_size=$INFERENCE_BATCH_SIZE \
    "dopamine_reward.perspectives=[$PERSPECTIVES]" \
    dopamine_reward.conda_env=$CONDA_ENV \
    rl_training.num_envs=$NUM_ENVS \
    rl_training.total_timesteps=$TOTAL_TIMESTEPS \
    rl_training.train_n_episodes=$TRAIN_EPISODES \
    rl_training.eval_n_episodes=$EVAL_EPISODES \
    "$@"

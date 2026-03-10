#!/bin/bash
# =============================================================================
# Robo-Dopamine Reward Model: Data Preparation + LoRA Finetuning + Merge + Eval
#
# This script runs the full pipeline to prepare a task-specific dense reward
# model for ManiFlow RL training. It runs in the robo-dopamine conda env.
#
# Usage:
#   conda activate robo-dopamine
#   bash scripts/prepare_and_finetune_dopamine.sh
#
# Prerequisites:
#   - robo-dopamine conda env with: torch 2.8, vLLM, peft, transformers, deepspeed
#   - MimicGen HDF5 dataset at DATASET_PATH
#   - Base model will be downloaded from HuggingFace on first run
# =============================================================================
set -e

# =====================================================
# Configuration — EDIT THESE FOR YOUR TASK
# =====================================================
TASK_NAME="nut_assembly_d0"
TASK_DESCRIPTION="assemble the round nut onto the correct peg"
DATASET_PATH="equi_diffpo/data/robomimic/datasets/${TASK_NAME}/${TASK_NAME}_voxel_abs.hdf5"
MAX_EPISODES=100          # Number of demo episodes to extract
MAX_SAMPLE_NUM=200        # Training pairs per episode (total ≈ MAX_EPISODES × MAX_SAMPLE_NUM)

BASE_MODEL="tanhuajie2001/Robo-Dopamine-GRM-3B"
ROBO_DOPAMINE_DIR="../../../Robo-Dopamine"  # Path to Robo-Dopamine repo (for training code)

# Derived paths (all absolute, CWD-independent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_PREP_DIR="$SCRIPT_DIR/dopamine_data_prep"
DATASET_PATH="$PROJECT_DIR/$DATASET_PATH"  # Resolve to absolute

# Data output paths (under Robo-Dopamine/dataset for training compatibility)
ROBO_DOPAMINE_ABS="$(cd "$SCRIPT_DIR/$ROBO_DOPAMINE_DIR" && pwd)"
RAW_OUTPUT_DIR="$ROBO_DOPAMINE_ABS/dataset/${TASK_NAME}_raw"
TRAIN_DATA_DIR="$ROBO_DOPAMINE_ABS/dataset/${TASK_NAME}_train"
TRAIN_DATA_LINK="$ROBO_DOPAMINE_ABS/dataset/train_data"  # Symlink for path compatibility
FINETUNE_JSON_DIR="$TRAIN_DATA_DIR/train_jsons"

# Checkpoint paths
LORA_OUTPUT_DIR="$PROJECT_DIR/checkpoints/${TASK_NAME}_grm_lora"
MERGED_OUTPUT_DIR="$PROJECT_DIR/checkpoints/${TASK_NAME}_grm_merged"

# LoRA hyperparameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=4

echo "=============================================="
echo " Robo-Dopamine Finetuning Pipeline"
echo "=============================================="
echo "  Task:        $TASK_NAME"
echo "  Description: $TASK_DESCRIPTION"
echo "  Dataset:     $DATASET_PATH"
echo "  Episodes:    $MAX_EPISODES"
echo "  Samples/ep:  $MAX_SAMPLE_NUM"
echo "  Base model:  $BASE_MODEL"
echo "  LoRA output: $LORA_OUTPUT_DIR"
echo "  Merged:      $MERGED_OUTPUT_DIR"
echo "=============================================="

# =====================================================
# Step 1: Extract frames from MimicGen HDF5
# =====================================================
echo ""
echo "=========================================="
echo "Step 1: Extract demo frames from HDF5"
echo "=========================================="

if [ -d "$RAW_OUTPUT_DIR" ] && [ "$(ls -A $RAW_OUTPUT_DIR/episode_* 2>/dev/null | head -1)" ]; then
    echo "Raw data already exists at $RAW_OUTPUT_DIR, skipping extraction."
    echo "  (Delete this directory to re-extract)"
else
    python "$DATA_PREP_DIR/extract_mimicgen_demos.py" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$RAW_OUTPUT_DIR" \
        --task_description "$TASK_DESCRIPTION" \
        --max_episodes $MAX_EPISODES
fi

# =====================================================
# Step 2: Generate training pairs (bin-sampling)
# =====================================================
echo ""
echo "=========================================="
echo "Step 2: Generate training pairs"
echo "=========================================="

# The generate script uses --base-dir as both input and output base.
# It reads sample_frames.json from each episode and writes train.json.
# But it expects sample_frames.json to exist. Since extract_mimicgen_demos.py
# doesn't generate it, we need to create it from the raw frames.

# Create sample_frames.json for each episode (list of all frame indices)
echo "Creating sample_frames.json for each episode..."
python3 -c "
import os, json, glob

raw_dir = '$RAW_OUTPUT_DIR'
for ep_dir in sorted(glob.glob(os.path.join(raw_dir, 'episode_*'))):
    cam_high = os.path.join(ep_dir, 'cam_high')
    if not os.path.isdir(cam_high):
        continue
    frames = sorted(glob.glob(os.path.join(cam_high, 'frame_*.jpg')))
    # Extract frame indices as zero-padded strings
    ids = [os.path.basename(f).replace('frame_', '').replace('.jpg', '') for f in frames]
    sample_json = os.path.join(ep_dir, 'sample_frames.json')
    with open(sample_json, 'w') as f:
        json.dump(ids, f, indent=4)
print(f'Created sample_frames.json for all episodes')
"

# Copy task_instruction.json to raw dir if not there
if [ ! -f "$RAW_OUTPUT_DIR/task_instruction.json" ]; then
    echo "[\"$TASK_DESCRIPTION\"]" > "$RAW_OUTPUT_DIR/task_instruction.json"
fi

# Run the training data generation
python "$DATA_PREP_DIR/1_generate_data.py" \
    --base-dir "$RAW_OUTPUT_DIR" \
    --score-bins 25 \
    --gap-bins 4 \
    --oversample-factor 100 \
    --zero-ratio 0.05 \
    --max_sample_num $MAX_SAMPLE_NUM

# =====================================================
# Step 3: Postprocess (merge + augment)
# =====================================================
echo ""
echo "=========================================="
echo "Step 3: Postprocess training data"
echo "=========================================="

# Copy the raw data to train directory (with train.json files generated in step 2)
if [ "$RAW_OUTPUT_DIR" != "$TRAIN_DATA_DIR" ]; then
    echo "Copying processed data to $TRAIN_DATA_DIR..."
    # Only copy if they're different dirs
    if [ ! -d "$TRAIN_DATA_DIR" ]; then
        cp -r "$RAW_OUTPUT_DIR" "$TRAIN_DATA_DIR"
    fi
fi

mkdir -p "$FINETUNE_JSON_DIR"

python "$DATA_PREP_DIR/2_postprocess_data.py" \
    --root-dir "$RAW_OUTPUT_DIR" \
    --merged-json "$FINETUNE_JSON_DIR/finetune_data_wo_replace.json" \
    --final-json "$FINETUNE_JSON_DIR/finetune_data_final.json" \
    --replace-prob 0.75

# =====================================================
# Step 4: Update symlink + register dataset
# =====================================================
echo ""
echo "=========================================="
echo "Step 4: Update data symlink for training"
echo "=========================================="

# The image paths in the JSON are like "train_data/episode_xxx/cam_high/frame_xxx.jpg"
# The data loader joins data_path + image_path => ./dataset/train_data/episode_xxx/...
# So we need: ./dataset/train_data -> ${TASK_NAME}_train (or _raw)
# But the train.json image paths use the base-dir name as prefix.
# Since we passed --base-dir $RAW_OUTPUT_DIR, the prefix is the raw dir basename.

# Fix: update the image path prefix in the JSON to match "train_data"
RAW_BASENAME=$(basename "$RAW_OUTPUT_DIR")
if [ "$RAW_BASENAME" != "train_data" ]; then
    echo "Fixing image path prefix: $RAW_BASENAME -> train_data"
    python3 -c "
import json

json_path = '$FINETUNE_JSON_DIR/finetune_data_final.json'
with open(json_path, 'r') as f:
    data = json.load(f)

for item in data:
    if 'image' in item and isinstance(item['image'], list):
        item['image'] = [p.replace('${RAW_BASENAME}/', 'train_data/', 1) for p in item['image']]

with open(json_path, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'Fixed {len(data)} items')
"
fi

# Update symlink: train_data -> current task's raw data
rm -f "$TRAIN_DATA_LINK"
ln -sf "${TASK_NAME}_raw" "$TRAIN_DATA_LINK"
echo "Symlink: $TRAIN_DATA_LINK -> ${TASK_NAME}_raw"

# Update __init__.py to point to the new dataset
echo "Updating dataset registration in Robo-Dopamine..."
python3 -c "
import re

init_path = '$ROBO_DOPAMINE_ABS/train/qwenvl/data/__init__.py'
with open(init_path, 'r') as f:
    content = f.read()

# Update MIMICGEN_GRM_FINETUNE annotation_path
old_pattern = r'(MIMICGEN_GRM_FINETUNE\s*=\s*\{[^}]*\"annotation_path\":\s*\")([^\"]+)(\")'
new_annotation = './dataset/${TASK_NAME}_train/train_jsons/finetune_data_final.json'
content = re.sub(old_pattern, r'\g<1>' + new_annotation + r'\3', content)

with open(init_path, 'w') as f:
    f.write(content)
print(f'Updated annotation_path to: {new_annotation}')
"

# =====================================================
# Step 5: LoRA Finetuning
# =====================================================
echo ""
echo "=========================================="
echo "Step 5: LoRA Finetuning"
echo "=========================================="

mkdir -p "$LORA_OUTPUT_DIR"

cd "$ROBO_DOPAMINE_ABS/train"
export PYTHONPATH="$(pwd)"
export WANDB_MODE=disabled

python qwenvl/train/train_qwen.py \
    --model_name_or_path $BASE_MODEL \
    --tune_mm_llm False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --use_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --dataset_use mimicgen_grm_finetune \
    --output_dir "$LORA_OUTPUT_DIR" \
    --cache_dir "./.cache" \
    --bf16 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --optim adamw_torch \
    --model_max_length 8192 \
    --data_flatten False \
    --data_packing False \
    --max_pixels 76800 \
    --min_pixels 12544 \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels 1304576 \
    --video_min_frame_pixels 200704 \
    --num_train_epochs $NUM_EPOCHS \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --eval_strategy "no" \
    2>&1 | tee "$LORA_OUTPUT_DIR/train.log"

cd "$PROJECT_DIR"

# =====================================================
# Step 6: Merge LoRA weights
# =====================================================
echo ""
echo "=========================================="
echo "Step 6: Merge LoRA weights"
echo "=========================================="

python "$DATA_PREP_DIR/merge_lora_weights.py" \
    --base_model $BASE_MODEL \
    --lora_path "$LORA_OUTPUT_DIR" \
    --output_path "$MERGED_OUTPUT_DIR"

# =====================================================
# Step 7: Visualize reward on demo episodes
# =====================================================
echo ""
echo "=========================================="
echo "Step 7: Visualize reward quality"
echo "=========================================="

GOAL_IMAGE="$RAW_OUTPUT_DIR/goal_image.png"
VIS_DIR="$PROJECT_DIR/vis_results/${TASK_NAME}"

for ep in episode_000 episode_025 episode_050; do
    EP_DIR="$RAW_OUTPUT_DIR/$ep"
    if [ -d "$EP_DIR" ]; then
        echo "--- Evaluating $ep (finetuned, forward mode) ---"
        python "$DATA_PREP_DIR/visualize_demo_reward.py" \
            --model_path "$MERGED_OUTPUT_DIR" \
            --demo_dir "$EP_DIR" \
            --goal_image "$GOAL_IMAGE" \
            --task "$TASK_DESCRIPTION" \
            --output_dir "$VIS_DIR/finetuned" \
            --eval_mode forward \
            --frame_interval 15 \
            --batch_size 4
    fi
done

echo ""
echo "=============================================="
echo " DONE!"
echo "=============================================="
echo "  Merged model:  $MERGED_OUTPUT_DIR"
echo "  Visualization: $VIS_DIR"
echo ""
echo "  Next: Run RL training with dense reward:"
echo "    bash scripts/run_rl_with_dopamine_reward.sh"
echo "=============================================="

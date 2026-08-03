#!/bin/bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

eval "$(conda shell.bash hook)"
# conda activate crf

BATCH_SIZE=4
CACHE_DIR="/workspace/.cache"
DATA_PATH="anonymousubmission/notes_with_related_items"
VAL_SPLIT="dev"
TEST_SPLIT="test"
MAX_N_EXAMPLES_VAL="None"
MAX_N_EXAMPLES_TEST="None"

setting_for_example_in_setting_3='setting_1'

CONFIGS=(
  "google/gemma-3-27b-it" 
) 

for config in "${CONFIGS[@]}"; do  echo "Zero-shot evaluation: ${config}"
  if ! nohup python e_zero_shot.py \
    --val_data_path "$DATA_PATH" \
    --val_data_split "$VAL_SPLIT" \
    --test_data_split "$TEST_SPLIT" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --task_type_list "setting_3" \
    --setting_for_example_in_setting_3 "$setting_for_example_in_setting_3" \
    --cache_dir "$CACHE_DIR" \
    --model_name_or_path "$config" &> "${config//\//_}_setting_3.log"; then
    echo "Zero-shot evaluation failed: ${config}. Continuing." >&2
  fi
done
echo "Zero-shot evaluation completed."
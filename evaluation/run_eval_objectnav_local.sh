#!/usr/bin/env bash
# =============================================================================
# ObjectNav (Object Goal Navigation, HM3D) evaluation launcher - Local GPU version
#
# Task family : Object Goal (move-only)
# Config      : config/objectnav_hm3d_local.yaml
# Entry script: evaluate_navigation.py (unified evaluator)
#
# Usage:
#     bash run_eval_objectnav_local.sh
#
# Runs on one GPU by default; adjust --nproc_per_node for multiple GPUs.
# =============================================================================

# Resolve relative paths from evaluation/, regardless of the caller working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# Activate conda environment
# source /root/miniconda3/etc/profile.d/conda.sh
conda activate ggnav

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# export MAGNUM_GPU_VALIDATION=OFF
# export MAGNUM_DISABLE_EXTENSIONS="GL_EXT_disjoint_timer_query GL_EXT_texture_filter_anisotropic"

# ---- Task-specific settings ----
TURN=500
TASK='object'
SPLIT="val"
CONFIG="config/objectnav_hm3d_local.yaml"

echo "CONFIG=${CONFIG}"
echo "TURN=${TURN}"
echo "SPLIT=${SPLIT}"

# ---- Model and paths ----
MID_RUN_NAME="GG-NAV/checkpoint-66000"
PLANNER_NAME='shortest_path'
MODEL_NAME="${MODEL_PATH:-../checkpoints/${MID_RUN_NAME}}"
PROCESSOR_PATH="${MODEL_NAME}"

# ---- Output directory ----
OUTPUT_DIR="../output_/${TASK}/${MID_RUN_NAME}_${TURN}/${SPLIT}"
mkdir -p "${OUTPUT_DIR}"

# ---- Run on one GPU via torchrun ----
torchrun --nproc_per_node=1 --master_port=29500 evaluate_navigation.py \
    --turn ${TURN} \
    --task ${TASK} \
    --base_url http://35.220.164.252:3888/v1 \
    --model_name gpt-4o \
    --openai_api_key api_key.txt \
    --model_path "${MODEL_NAME}" \
    --processor_path "${PROCESSOR_PATH}" \
    --local_planner ${PLANNER_NAME} \
    --output_path "${OUTPUT_DIR}" \
    --resize_h 384 \
    --resize_w 384 \
    --habitat_config_path ${CONFIG} \
    --baseline_config_path config/gen_videos.yaml \
    --eval_split ${SPLIT} \
    --scene_summary ../data/scene_summary \
    # --save_video

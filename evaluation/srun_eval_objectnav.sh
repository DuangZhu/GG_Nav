#!/usr/bin/env bash
# ObjectNav (Object Goal Navigation, HM3D) evaluation launcher.
# Task family : Object Goal (move-only)
# Config      : config/objectnav_hm3d.yaml
# Entry script: evaluate_navigation.py (unified evaluator)
# Usage:
#     bash srun_eval_objectnav.sh
# NOTE: configure model checkpoints and API settings before running;
#       place checkpoints under ../checkpoints/ or set MODEL_PATH for both model and processor.
#       Data paths resolve under ../data/.

# Resolve relative paths from evaluation/, regardless of the caller working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

# ---- Task-specific settings ----
TURN=500
TASK='object'
SPLIT="val"
CONFIG="config/objectnav_hm3d.yaml"

echo "CONFIG=${CONFIG}"
echo "TURN=${TURN}"
echo "SPLIT=${SPLIT}"

MID_RUN_NAME="20260416_one_step_two_task/checkpoint-66000"
PLANNER_NAME='shortest_path'
MODEL_NAME="${MODEL_PATH:-../checkpoints/${MID_RUN_NAME}}"
PROCESSOR_PATH="${MODEL_NAME}"

srun -p wam_agent \
    --gres=gpu:8 \
    --ntasks=8 \
    --time=0-20:00:00 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --exclude=HOST-10-140-60-26,HOST-10-140-60-173 \
    python -u evaluate_navigation.py \
    --turn ${TURN} \
    --task ${TASK} \
    --base_url http://35.220.164.252:3888/v1 \
    --model_name gpt-4o \
    --openai_api_key api_key.txt \
    --model_path "${MODEL_NAME}" \
    --processor_path "${PROCESSOR_PATH}" \
    --local_planner ${PLANNER_NAME} \
    --output_path "../output/${TASK}/${MID_RUN_NAME}_${TURN}/${SPLIT}" \
    --resize_h 384 \
    --resize_w 384 \
    --habitat_config_path ${CONFIG} \
    --baseline_config_path config/gen_videos.yaml \
    --eval_split ${SPLIT} \
    --scene_summary ../data/scene_summary \

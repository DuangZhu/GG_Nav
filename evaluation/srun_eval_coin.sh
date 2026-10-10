#!/usr/bin/env bash
# CoIN (Conversational Instance Navigation, HM3D) evaluation launcher.
# Task family : IIGN (CoIN is an IIGN sub-type; instruction is built from
#               ``episode.object_category + " | " + episode.instruction``,
#               and the agent is allowed to both move and talk)
# Config      : config/iign_coin.yaml
# Entry script: evaluate_navigation.py (unified evaluator)
# Usage:
#     bash srun_eval_coin.sh
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
TURN=62
TASK='instance_dialog'
SPLIT="val_seen_synonyms"
CONFIG="config/iign_coin.yaml"

echo "CONFIG=${CONFIG}"
echo "TURN=${TURN}"
echo "SPLIT=${SPLIT}"

# ---- Model / planner ----
MID_RUN_NAME="20260416_one_step_two_task/checkpoint-54000"
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
    --dialog_enabled \
    --scene_summary ../data/scene_summary \
    --dialog_history_mode recent_dialogs \
    --dialog_recent_step_n 4000 \
    --dialog_recent_n 20 \
    --dialog_radius 240.0 \

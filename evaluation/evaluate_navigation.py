"""Unified release-time entrypoint for navigation evaluation.

This script is a thin wrapper shared by every navigation task. It only
handles:

  * command line parsing (union of the flags accepted by every task);
  * distributed initialisation via :func:`init_distributed_mode`;
  * loading of the primary VLM policy (``Qwen3VLForConditionalGeneration``)
    for ``shortest_path`` evaluation;
  * optional loading of a second detector model driven by
    ``--oracle_detector_model_path``;
  * dispatching to the correct task-specific ``VLNEvaluator`` through
    :func:`tasks.get_evaluator` (Object Goal vs. IIGN);
  * ``dist.all_reduce`` aggregation of per-rank success / SPL / OS / NE
    counters and persisting the ``result.json`` / ``result_all.json`` output.

No ``VLNEvaluator`` class is defined here on purpose - keep the file small so
that new users can grasp the launcher in one glance.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dist import init_distributed_mode, get_rank, get_world_size
from npc.simple_npc.simple_npc import SimpleNPC
from tasks import get_evaluator


# CLI parsing
def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map the ``--dtype`` string to a concrete ``torch.dtype``."""
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    return torch.bfloat16


def parse_args() -> argparse.Namespace:
    """Build the argparse namespace shared by every evaluation task.

    The flag set is the *union* of what every task accepts, keeping every
    name / default / choice list so that the existing ``srun_eval_*.sh``
    scripts run unchanged.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--oracle_detector_model_path", type=str, default="")
    parser.add_argument(
        "--processor_path",
        type=str,
        default="",
        help="Processor checkpoint; defaults to --model_path.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--dialog_enabled", action="store_true", default=False)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--baseline_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--local_planner", type=str, default="shortest_path", choices=["shortest_path"])
    parser.add_argument("--openai_api_key", type=str, default="habitat/npc/api_key/api_key.txt")
    parser.add_argument("--base_url", type=str, default="localhost:0.0.0.0")
    parser.add_argument("--task", type=str, default="instance")
    parser.add_argument("--scene_summary", type=str, default="data/scene_summary")
    parser.add_argument("--output_path", type=str, default="./exps_pix/val_unseen/debug_coord_wm")
    parser.add_argument("--turn", type=int, default=5)
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--use_prompt_image", action="store_true", default=False)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # Distributed knobs consumed by ``init_distributed_mode``.
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank")
    parser.add_argument("--gpu", default=0, type=int, help="gpu")
    parser.add_argument("--port", default="2334")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    # Dialog-history related flags (used by the IIGN evaluator).
    parser.add_argument(
        "--dialog_history_mode",
        type=str,
        default="all",
        choices=["all", "recent_steps", "recent_dialogs", "spatial_local"],
    )
    parser.add_argument("--dialog_recent_step_n", type=int, default=8)
    parser.add_argument("--dialog_recent_n", type=int, default=8)
    parser.add_argument("--dialog_radius", type=float, default=8.0)

    return parser.parse_args()


# Model loading
def _load_models(args: argparse.Namespace, device: torch.device):
    """Load the primary policy and, when required, the auxiliary detector.

    Mirrors the model-loading blocks that used to live at the top of ``eval()``
    inside both legacy scripts.
    """
    model_dtype = _resolve_dtype(args.dtype)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        device_map={"": device},
    )

    detector_path = args.oracle_detector_model_path.strip()
    if detector_path:
        model_2 = Qwen3VLForConditionalGeneration.from_pretrained(
            detector_path,
            dtype=model_dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    else:
        # Reuse the same backbone to reduce VRAM when detector checkpoint is
        # not separately provided.
        model_2 = model

    return model, model_2


# Distributed metric aggregation
def _sum_or_zero(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    return x.float().sum()


def _aggregate_and_dump(sucs, spls, oss, nes, args: argparse.Namespace) -> None:
    """All-reduce per-rank tensors and persist the aggregated JSON reports."""
    local_count = torch.tensor(float(sucs.numel()), device=sucs.device)
    local_sucs = _sum_or_zero(sucs)
    local_spls = _sum_or_zero(spls)
    local_oss = _sum_or_zero(oss)
    local_nes = _sum_or_zero(nes)

    if dist.is_available() and dist.is_initialized():
        for t in [local_count, local_sucs, local_spls, local_oss, local_nes]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_count = max(local_count.item(), 1.0)
    result_all = {
        "sucs_all": (local_sucs / total_count).item(),
        "spls_all": (local_spls / total_count).item(),
        "oss_all": (local_oss / total_count).item(),
        "nes_all": (local_nes / total_count).item(),
        "length": int(local_count.item()),
    }
    print(result_all)

    if get_rank() == 0:
        with open(os.path.join(args.output_path, "result_all.json"), "w", encoding="utf-8") as file:
            json.dump(result_all, file, indent=4, ensure_ascii=False)
        with open(os.path.join(args.output_path, "result.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result_all, ensure_ascii=False) + "\n")


# Top-level orchestration
def evaluate(model, processor, args: argparse.Namespace, model_2=None) -> None:
    """Instantiate the task-specific evaluator and run one full pass."""
    model.eval()
    if model_2 is not None and model_2 is not model:
        model_2.eval()

    simple_npc = SimpleNPC(
        max_interaction_turn=10,
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        base_url=args.base_url,
    )

    # Dispatch to the correct ``VLNEvaluator`` implementation. ``dialog_enabled``
    # takes precedence over ``--task`` to mirror the legacy behaviour.
    EvaluatorCls = get_evaluator(args.task, args.dialog_enabled)
    evaluator = EvaluatorCls(
        npc=simple_npc,
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=get_world_size(),
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
        model_2=model_2,
    )
    sucs, spls, oss, nes, _ = evaluator.eval_action(idx=get_rank())
    _aggregate_and_dump(sucs, spls, oss, nes, args)


def main() -> None:
    args = parse_args()
    init_distributed_mode(args)
    os.makedirs(args.output_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path)
    processor.tokenizer.padding_side = "left"
    device = torch.device(f"cuda:{args.local_rank}")

    model, model_2 = _load_models(args, device)
    evaluate(model, processor, args, model_2)


if __name__ == "__main__":
    main()

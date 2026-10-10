"""Shared helpers used by both objectnav and iign task evaluators.

This module is internal to `evaluation.tasks` and is not intended to be imported by
end users. Symbols here are the byte-for-byte common subset that was previously
duplicated in `tasks/objectnav.py` and `tasks/iign.py`.
"""

import os
import json
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple, Any, Union, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import quaternion
import cv2
from pathlib import Path
import random
import re
import torch
import habitat_sim
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import OmegaConf, DictConfig, open_dict
from npc.utils.get_description import get_path_description, get_path_description_without_additional_info
from dist import *
from qwenvl.data.utils.pair_question import match_utterance
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class EpisodeMeta:
    scene_id: str
    episode_id: int
    episode_instruction: str
    instance_type: str


def _extract_text_from_messages(x: Any) -> str:
    """
    Recursively collect all text fields from messages_track; ignore images.
    """
    texts = []

    def _walk(obj: Any):
        if obj is None:
            return

        if isinstance(obj, str):
            texts.append(obj)
            return

        if isinstance(obj, dict):
            if obj.get("type") == "text" and isinstance(obj.get("text"), str):
                texts.append(obj["text"])
                return

            for v in obj.values():
                _walk(v)
            return

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _walk(item)
            return

        return

    _walk(x)
    return "\n".join(texts)


def add_more_new_instance(
    label_set: Set[str],
    llm_outputs: str,
    messages_track: Any,
    instance_type: str = "",
) -> Set[str]:
    """
    Add instance labels present in llm_outputs but missing from messages_track to label_set.
    Only labels of the form {instance_type}_\d+ (e.g. chair_26) are considered.
    """
    history_text = _extract_text_from_messages(messages_track[0]) # hardcode

    history_labels = set(extract_typed_instances(history_text, instance_type))
    llm_labels = set(extract_typed_instances(llm_outputs, instance_type))

    new_labels = llm_labels - history_labels
    label_set.update(new_labels)
    return label_set


def get_config(
    habitat_config_path: str,
    baseline_config_path: str,
    opts: Optional[list] = None,
    configs_dir: str = os.path.dirname(inspect.getabsfile(inspect.currentframe())),
) -> DictConfig:
    """
    Returns habitat_baselines config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat_baselines.trainer_name=ddppo"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_BASELINES_CFG_DIR`).
    :return: composed config object.
    """
    habitat_config = get_habitat_config(
        habitat_config_path, overrides=opts, configs_dir=configs_dir
    )
    baseline_config = OmegaConf.load(baseline_config_path)
    
    with open_dict(habitat_config):
        config = OmegaConf.merge(habitat_config, baseline_config)
    
    return config


def split_and_clean(text):
    import re
    parts = re.split(r'(<image>)', text)
    results = []
    for part in parts:
        if part == '<image>':
            results.append(part)
        else:
            # if '<think>' not in part:
            clean_part = part.replace('\n', '').strip()
            # else:
            #     clean_part = part
            if clean_part:
                results.append(clean_part)
    return results


def ordinals_clause(indices: List[int]) -> str:
    """
    Convert a list of ints into an ordinal-word clause.

    Rules:
    - If the list has 1 element: return "first"
    - If it has 2 elements: return "first or second"
    - If it has 3+ elements: return "first, second, third or fourth" (Oxford comma style)

    Supports values in [1, 20].
    """
    if not isinstance(indices, list):
        raise TypeError(f"indices must be a list, got {type(indices).__name__}")
    if len(indices) == 0:
        raise ValueError("indices must be non-empty")

    for x in indices:
        if not isinstance(x, int):
            raise TypeError(f"all elements must be int, got {type(x).__name__}")
        if x < 0 or x > 20:
            raise ValueError(f"all elements must be in [1, 20], got {x}")

    ordinal_map = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    }
    words = [ordinal_map[x+1] for x in indices]
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} or {words[1]}"
    return f"{', '.join(words[:-1])} or {words[-1]}"


def idx_to_ordinal_word(idx: int) -> str:
    """0-based idx -> 'first', 'second', ... (no digits)."""
    words = [
        "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
        "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth","twentieth"
    ]
    idx = int(idx)
    if idx < 0 or idx >= len(words):
        raise ValueError(f"idx out of range: {idx} (supported 0~{len(words)-1})")
    return words[idx]


def save_imgs(messages, out_dir, prefix="img"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    k = 0
    for m in messages:
        for it in m.get("content", []):
            if it.get("type") == "image" and isinstance(it.get("image"), Image.Image):
                it["image"].save(out / f"{prefix}_{k:06d}.png")
                k += 1


def calculate_path_length(path):
    accumulated_length = [0]
    for i, p in enumerate(path[1:]):
        accumulated_length.append(accumulated_length[i] + np.linalg.norm(np.array(p) - np.array(path[i])))
    return accumulated_length


def get_shortest_path(env, start_position, target_position):
    """
    Find the shortest path from the current position to the target position in a habitat env.

    Args:
        env: habitat environment instance
        start_position: start coordinate as numpy array [x, y, z]
        target_position: target coordinate as numpy array [x, y, z]

    Returns:
        path: list of waypoints
        success: whether a valid path was found
    """
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = target_position
    
    success = env.sim.pathfinder.find_path(
        shortest_path
    )
    return shortest_path.points, success


def find_order(question: str, default: int = 0, max_n: int = 20) -> int:
    """Find English ordinal word in question, return 0-based index (first->0 ...)."""
    q = (question or "").lower()

    ord_words = [
        "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
        "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth","twentieth"
    ][:max_n]

    for i, w in enumerate(ord_words):
        if re.search(rf"\b{w}\b", q):
            return i

    return default


def IoU(boxA, boxB):
    """
    box: [x1, y1, x2, y2] (pixel coords)
    return IoU in [0,1]
    """
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    # intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    # areas
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter
    return 0.0 if union <= 0 else inter / union


def get_navigable_path(env, start_position, target_positions: list, object_info: dict):
    start_position = [float(i) for i in start_position]
    target_positions = sorted(target_positions, key=lambda x: np.linalg.norm(np.array(x['agent_state']['position']) - np.array(object_info['position'])))
    success = False
    while not success and len(target_positions) > 0:
        target_position = target_positions.pop(0)
        shortest_path, success = get_shortest_path(env, start_position, target_position['agent_state']['position'])
    if success:
        return shortest_path, True
    else:
        return [], False


def get_path_description_(env, object_dict, region_dict):
    goal_path, success = get_navigable_path(env, env.sim.get_agent_state().position, [{'agent_state': {'position': vp.agent_state.position}} for vp in env.current_episode.goals[0].view_points], {'position': env.current_episode.goals[0].position})
    if not success or len(np.unique(goal_path, axis=0)) == 1:
        print('no shortest path')
        return None, 0
    path_length = calculate_path_length(goal_path)
    pl = path_length[-1]
    goal_index = max([i for i, c in enumerate(path_length) if c < 4])
    # goal_index = len(goal_path)-1
    if goal_index == 0:
        goal_index = len(goal_path)-1
    questioned_path = goal_path[:goal_index+1]
    current_yaw = 2 * np.arctan2(env.sim.get_agent_state().rotation.y, env.sim.get_agent_state().rotation.w)
    _, idx = np.unique(questioned_path, axis=0, return_index=True)
    idx_sorted = np.sort(idx)
    questioned_path = list(np.array(questioned_path)[idx_sorted])
    try:
        path_description, _ = get_path_description(quaternion.from_euler_angles([0, current_yaw, 0]), questioned_path, object_dict, region_dict, return_finish=False, height_list=[env.sim.get_agent_state().position[1]]*len(questioned_path))
    except:
        path_description, _ = get_path_description_without_additional_info(quaternion.from_euler_angles([0, current_yaw, 0]), questioned_path, height_list=[env.sim.get_agent_state().position[1]]*len(questioned_path))
    return path_description, pl


def open_image(image_or_image_path):
    if isinstance(image_or_image_path, Image.Image):
        return image_or_image_path
    elif isinstance(image_or_image_path, str):
        return Image.open(image_or_image_path)
    else:
        raise ValueError("Unsupported input type!")


def get_bbox_from_mask(mask):
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return [-1,-1,-1,-1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def unify_to_first(
    vis_frames,
    method: str = "resize",
    pad_color=(0, 0, 0),
    assume_rgb: bool = True
):
    assert len(vis_frames) > 0, "vis_frames 为空"
    h0, w0 = vis_frames[0].shape[:2]
    out = []

    for i, f in enumerate(vis_frames):
        f = np.asarray(f)

        if f.ndim == 2:
            f = np.stack([f]*3, axis=2)
        if f.shape[2] > 3:
            f = f[:, :, :3]

        if f.dtype != np.uint8:
            fmax = float(np.nanmax(f)) if f.size else 1.0
            f = (f * 255.0) if fmax <= 1.5 else np.clip(f, 0, 255)
            f = f.astype(np.uint8)

        h, w = f.shape[:2]
        if (h, w) == (h0, w0):
            out.append(np.ascontiguousarray(f))
            continue

        if method == "letterbox":
            scale = min(w0 / w, h0 / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            canvas = np.full((h0, w0, 3), pad_color, dtype=np.uint8)
            top, left = (h0 - nh) // 2, (w0 - nw) // 2
            canvas[top:top+nh, left:left+nw] = resized
            f_out = canvas
        else:
            f_out = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA if (h*w) > (h0*w0) else cv2.INTER_LINEAR)

        out.append(np.ascontiguousarray(f_out))

    return out


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Extract the JSON list inside a fenced code block like:
    ```json
    [ ... ]
    ```
    Return it as a Python list of dicts.
    """
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("No fenced JSON code block found (```json ... ```).")

    payload = m.group(1).strip()

    # If the block contains extra text, try to isolate the first [...] segment
    b = re.search(r"\[[\s\S]*\]", payload)
    if not b:
        raise ValueError("No JSON list found inside the code block.")
    payload = b.group(0)

    data = json.loads(payload)
    if not isinstance(data[0], dict):
        bbox_tuples = re.findall(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            text
        )
        if not bbox_tuples:
            raise ValueError("No bbox like [x1,y1,x2,y2] found in text.")

        labels = re.findall(r'label\s*["\']?\s*:\s*["\']([^"\']+_\d+)["\']', text, flags=re.IGNORECASE)
        if not labels:
            labels = re.findall(r"\b([A-Za-z0-9]+_\d+)\b", text)

        out = []
        for i, (x1, y1, x2, y2) in enumerate(bbox_tuples):
            bbox = [int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))]
            label = labels[i] if i < len(labels) else f"chair_{i}"
            out.append({"bbox_2d": bbox, "label": label})
        return out
    else:
        return data


def denorm_bbox(bbox_list, image, scale=1000.0, clamp=True):
    """
    Only supports the format:
    [{'bbox_2d': [x1, y1, x2, y2], 'label': 'refrigerator_0'}, ...]
    bbox_2d is expected to be normalized in [0, scale] (default 1000); denormalize to pixel coordinates.
    """
    if isinstance(image, np.ndarray):
        W, H = image.shape[1], image.shape[0]
    else:
        W, H = image.size
    out = []

    for item in bbox_list:
        x1, y1, x2, y2 = item["bbox_2d"]
        x1 = x1 / scale * W
        x2 = x2 / scale * W
        y1 = y1 / scale * H
        y2 = y2 / scale * H

        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        if clamp:
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))

        out.append({
            "bbox_2d": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "label": item["label"],
        })

    return out


def mask01_to_bbox(mask):
    """
    mask: HxW binary map with values 0/1 (numpy or torch).
    Returns [x1, y1, x2, y2], or None if no positive pixel exists.
    """
    if torch.is_tensor(mask):
        ys, xs = torch.where(mask == 1)
        if ys.numel() == 0:
            return None
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        return [x1, y1, x2, y2]
    else:
        mask = np.asarray(mask)
        ys, xs = np.where(mask == 1)
        if xs.size == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def max_label_id(dialogs, instance_type):
    pat = re.compile(rf"{re.escape(instance_type)}_(\d+)")
    m = [int(x) for d in dialogs if d.get("role")=="navigator"
         for x in pat.findall(d.get("thinking",""))]
    return max(m, default=-1)


def draw_bboxes(img: Image.Image, items, is_norm: bool = True, gt_bbox = None) -> Image.Image:
    """
    img: PIL.Image
    items: list[{"bbox_2d":[x1,y1,x2,y2], "label":...}, ...]
    is_norm:
        - True: bbox coords are in a 1000x1000 reference space -> scale to img size
        - False: bbox coords are already in img pixel space
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    W, H = out.size
    sx, sy = (W / 1000.0, H / 1000.0) if is_norm else (1.0, 1.0)

    for it in items:
        x1, y1, x2, y2 = it["bbox_2d"]
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        label = str(it.get("label", ""))
        if label:
            tw, th = draw.textsize(label, font=font)
            draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 4, y1], fill=(255, 0, 0))
            draw.text((x1 + 2, max(0, y1 - th - 2)), label, fill=(255, 255, 255), font=font)
    
    if gt_bbox:
        sx, sy = (W / 640.0, H / 480.0) 
        x1, y1, x2, y2 = gt_bbox
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        return out, [x1, y1, x2, y2]
    else:
        return out, None


def ask_npc_answer(prompt="NPC answer> ", default=""):
    ans = input(prompt).strip()
    return ans if ans else default


def format_with_tabs(data_list: List[Any]) -> str:
    formatted_lines = [
        f"\t{json.dumps(item, ensure_ascii=False, separators=(',', ':'))}"
        for item in (data_list or [])
    ]
    content = ",\n".join(formatted_lines)
    json_block = f"```json\n[\n{content}\n]\n```"
    return f"{json_block}\n"


def grounding_npc_answer(
    num_unseen: int,
    question: str,
    segment,          # 0/1 mask, observations['semantic'][0]
    bbox_list,        # [{'bbox_2d':[...], 'label':...}, ...] (norm 0~1000)
    image,            # cur_images[0]
    iou_thr: float = 0.1,
    out_image_path = None
):  
    gt_box = mask01_to_bbox(segment)
    denorm_bbox_list = denorm_bbox(bbox_list, image, scale=1000.0, clamp=True)
    bbox_image, gt_box = draw_bboxes(image, bbox_list, is_norm=True, gt_bbox=gt_box)
    p = Path(out_image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bbox_image.save(out_image_path)
    
    def yes_no_by_iou(pred_box):
        if not gt_box or pred_box is None:
            return "No."
        return "Yes." if IoU(gt_box, pred_box) > iou_thr else "No."

    if num_unseen == 1:
        if not denorm_bbox_list:
            return "No."

        if len(denorm_bbox_list) == 1:
            return yes_no_by_iou(denorm_bbox_list[0]["bbox_2d"])
        q_res = match_utterance(question, TEMPLATES_TEST) # 0-based int
        bbox = next(item["bbox_2d"] for item in denorm_bbox_list if item["label"] == q_res['slots']['except_clause'])
        return yes_no_by_iou(bbox)

    if num_unseen > 1:
        if not gt_box or not denorm_bbox_list:
            return "None of them."
        ious = [IoU(gt_box, item["bbox_2d"]) for item in denorm_bbox_list]
        best_idx = max(range(len(ious)), key=ious.__getitem__)
        best_score = ious[best_idx]

        if best_score > iou_thr:
            return f"{denorm_bbox_list[best_idx]['label']}."
        return "None of them."
    # if num_unseen == -1:
    #     return ''
    raise ValueError(f"num_unseen invalid (expect >=1). got {num_unseen}")


def split_llm_output(llm_outputs: str) -> Tuple[Optional[str], str]:
    s = llm_outputs.strip()

    m_think = re.search(r"(<think>[\s\S]*?</think>)", s)
    think = m_think.group(1) if m_think else None

    m_talk = re.search(r"(<talk>[\s\S]*)$", s)
    question = m_talk.group(1) if m_talk else None
    if question:
        question = question.replace('<talk>', '').strip()

    return think, question


def traj_to_actions(dp_actions, use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory


def _safe_load_json(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            import ast
            return ast.literal_eval(s)
        except Exception:
            return None


def extract_bbox_blocks_from_chat_sources(msg):
    """
    Returns a list where each element is the detection list for one frame:
    [
      [ {"bbox_2d":[...],"label":"chair_0"}, ... ],  # frame0
      [ {"bbox_2d":[...],"label":"chair_4"}, ... ],  # frame1
      ...
    ]
    Bboxes are clipped to [0, 1000].
    """
    def clamp_bbox(bb, min_v=0, max_v=1000):
        out = []
        for x in bb:
            try:
                x = int(x)
            except Exception:
                return None
            x = max(min_v, min(max_v, x))
            out.append(x)
        return out

    blocks = []
    text = msg
    for m in re.finditer(r"```json\s*([\s\S]*?)\s*```", text):
        data = _safe_load_json(m.group(1))
        if not isinstance(data, list) or len(data) == 0:
            blocks.append([])
            continue

        cleaned = []
        ok = True
        for it in data:
            if not (isinstance(it, dict) and "bbox_2d" in it and "label" in it):
                ok = False
                break

            bb = it.get("bbox_2d")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                ok = False
                break

            bb = clamp_bbox(bb)
            if bb is None:
                ok = False
                break

            cleaned.append({
                "bbox_2d": bb,
                "label": it["label"]
            })

        if ok:
            blocks.append(cleaned)

    return blocks


def draw_bboxes_from_bbox_json(img, dets, width=3):
    """dets: [{"bbox_2d":[x1,y1,x2,y2],"label":"xxx"}, ...]"""
    if not dets:
        return img
    d = ImageDraw.Draw(img)
    w, h = img.size

    for obj in dets:
        label = str(obj.get("label", ""))
        bbox = obj.get("bbox_2d", None)
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        mx = max(bbox)

        if mx <= 1.5:
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
        elif mx <= 1000 and (mx > w or mx > h):
            x1, x2 = int(x1 / 1000 * w), int(x2 / 1000 * w)
            y1, y2 = int(y1 / 1000 * h), int(y2 / 1000 * h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 255, 0)
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if label:
            tb = d.textbbox((0, 0), label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = 2
            bx1, by1 = x1, max(0, y1 - th - 2 * pad)
            bx2, by2 = x1 + tw + 2 * pad, by1 + th + 2 * pad
            d.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0))
            d.text((bx1 + pad, by1 + pad), label, fill=color)

    return img


def save_seg_map_bw(seg_map: np.ndarray, out_path: str):
    # seg_map: bool/0-1 array, shape (H, W)
    m = (seg_map.astype(np.uint8) * 255)  # False->0, True->255
    Image.fromarray(m, mode="L").save(out_path)


def remove_checked_summary(text: str) -> str:
    pat = r"\s*\d+\s+of them have been checked and are not the target instance\.\s*The remaining\s+\d+\s+are still uncertain\.\s*"
    text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text_list = text.split('```')
    text_list[-1] = '\n'+text_list[-1]
    text = '```'.join(text_list)
    return text


def extract_typed_instances(text: str, instance_type: str, unique: bool = True) -> List[str]:
    """
    Extract all strings of the form "{instance_type}_{int}" from text.
    - instance_type may contain a space or underscore (e.g. "washing machine" / "washing_machine").
    - Order-preserving dedup by default; if unique=False, duplicates are kept.
    """
    parts = [re.escape(p) for p in re.split(r"[ _]+", instance_type.strip()) if p]
    if not parts:
        return []
    type_pat = r"(?:%s)" % r"[ _]+".join(parts)

    pat = re.compile(rf"\b({type_pat}_\d+)\b", flags=re.IGNORECASE)
    hits = pat.findall(text)

    if not unique:
        return hits
    return list(dict.fromkeys(hits))


def match_labels_min_frames(
    bbox_blocks: List[List[Dict[str, Any]]],
    label_set: Set[str],
    pre_decision_step: int,
    max_frames: int = 8,
) -> Dict[str, Tuple[int, Dict[str, Any]]]:
    """
    Cover every label in label_set with as few frames as possible (when total frames <= max_frames,
    enumerate subsets to obtain the exact optimum).
    Returns matched: label -> (abs_frame_idx, item)
    """
    def popcount(x: int) -> int:
        return bin(x).count("1")

    F = len(bbox_blocks)
    if not label_set or F == 0:
        return {}

    if F > max_frames:
        offset = F - max_frames
        bbox_blocks = bbox_blocks[offset:]
        pre_decision_step += offset
        F = max_frames

    labels = list(label_set)
    idmap = {lb: i for i, lb in enumerate(labels)}
    L = len(labels)
    target = (1 << L) - 1

    fmask = [0] * F
    frep: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(F)]
    for fi, items in enumerate(bbox_blocks):
        m = 0
        rep = frep[fi]
        for it in items:
            lb = it.get("label")
            if lb in idmap:
                m |= 1 << idmap[lb]
                rep.setdefault(lb, it)
        fmask[fi] = m

    best_s = 0
    best_key = None  # (ok?, cnt, -score, -covered, s)
    for s in range(1, 1 << F):
        cnt = popcount(s)
        m = 0
        score = 0
        for fi in range(F):
            if (s >> fi) & 1:
                m |= fmask[fi]
                score += fi
        mm = m & target
        ok = (mm == target)
        covered = popcount(mm)
        key = (0 if ok else 1, cnt, -score, -covered, s)
        if best_key is None or key < best_key:
            best_key = key
            best_s = s

    selected = [fi for fi in range(F) if (best_s >> fi) & 1]

    matched: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for fi in sorted(selected, reverse=True):
        abs_fi = pre_decision_step + fi
        for lb, it in frep[fi].items():
            if lb in label_set and lb not in matched:
                matched[lb] = (abs_fi, it)
    return matched


def save_dialog_bbox_track(
    rgb_list: List[Union[str, Image.Image]],
    dialogs: List[Dict[str, Any]],
    out_dir: str,
    prefix: str = "step",
    only_navigator: bool = False,
) -> List[str]:
    """
    Pick frames from rgb_list according to dialogs[i]["true_idx"], draw the bbox+label from
    the thinking section on each frame, and save the images.

    Depends on the following existing helpers:
      - open_image(image_or_image_path)
      - extract_bbox_blocks_from_chat_sources(msg)
      - extract_json_list(text)
      - draw_bboxes_from_bbox_json(img, dets, width=3)

    Returns: list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, dialog in enumerate(dialogs):
        role = dialog.get("role", "unknown")
        if only_navigator and role != "navigator":
            continue

        idx = dialog.get("true_idx", None)
        if idx is None:
            continue
        idx = int(idx)
        if idx < 0 or idx >= len(rgb_list):
            continue

        img = open_image(rgb_list[idx]).convert("RGB")

        thinking = dialog.get("thinking", "") or ""
        msg = dialog.get("message", "") or ""

        dets = []
        try:
            blocks = extract_bbox_blocks_from_chat_sources(thinking)
            if blocks:
                dets = blocks[-1]
        except Exception:
            dets = []

        if not dets:
            try:
                dets = extract_json_list(thinking)
            except Exception:
                try:
                    dets = extract_json_list(msg)
                except Exception:
                    dets = []

        if dets:
            img_drawn = draw_bboxes_from_bbox_json(img.copy(), dets, width=3)
        else:
            img_drawn = img

        out_path = os.path.join(out_dir, f"{prefix}_{i:04d}_{role}_idx{idx:04d}.png")
        img_drawn.save(out_path)
        saved.append(out_path)

    return saved


def make_questions_to_list(questions, L):
    result = [None] * L
    pattern = r"Frame_(\d+):\s*(.*?)(?=\s*Frame_\d+:|$)"

    matches = re.findall(pattern, questions)

    for frame_idx, text in matches:
        idx = int(frame_idx)-1
        if 0 <= idx < L:
            result[idx] = text.strip()

    return result


def get_last_target_label_from_dialogs(dialogs, instance_type: str) -> Optional[str]:
    """
    Recover the most recently confirmed target label from the dialog history.
    Rules:
    1) If the oracle explicitly answered with something like 'chair_3.', use it directly.
    2) If the oracle answered "Yes." and the corresponding navigator thinking contains exactly
       one instance label, use that label.
    """
    for i in range(len(dialogs) - 1, 0, -1):
        oracle = dialogs[i]
        nav = dialogs[i - 1]

        if oracle.get("role") != "oracle" or nav.get("role") != "navigator":
            continue

        oracle_msg = str(oracle.get("message", "") or "")
        nav_think = str(nav.get("thinking", "") or "")

        labels = extract_typed_instances(oracle_msg, instance_type)
        if labels:
            return labels[-1]

        if re.search(r"\byes\b", oracle_msg, flags=re.IGNORECASE):
            think_labels = extract_typed_instances(nav_think, instance_type)
            think_labels = list(dict.fromkeys(think_labels))
            if len(think_labels) == 1:
                return think_labels[0]

    return None


def build_updated_tracking_dialog(
    instance_type: str,
    bbox: List[Dict[str, Any]],
    target_name: str,
    abs_frame_idx: int,
):
    """
    Rebuild a dialog pair based on the bboxes in the newest frame.
    - Single target: single_candidate / Yes.
    - Multiple targets: multi_all_unseen / target_label.
    """
    bbox = sorted(bbox, key=lambda d: d["bbox_2d"][0])
    num_bbox = len(bbox)
    bbox_message = format_with_tabs(bbox)

    if num_bbox <= 1:
        question = random.choice(TEMPLATES_TEST[0]["question"]).format(instance=instance_type)
        npc_answer = "Yes."
    else:
        question = random.choice(TEMPLATES_TEST[1]["question"]).format(instance=instance_type)
        npc_answer = f"{target_name}."

    think = (
        f"<think>I am looking for a specific {instance_type}. "
        f"In this image, I detected {num_bbox} {instance_type}(s).\n"
        f"{bbox_message}"
        f" 0 of them have been checked and are not the target instance."
        f" The remaining {num_bbox} are still uncertain.</think>"
    )

    nav_dialog = {
        "role": "navigator",
        "message": question,
        "true_idx": abs_frame_idx,
        "thinking": think,
    }
    oracle_dialog = {
        "role": "oracle",
        "message": npc_answer,
        "true_idx": abs_frame_idx,
    }
    return nav_dialog, oracle_dialog


def visualize_gt_and_pred_boxes(
    image,
    gt_box: Optional[List[int]],
    pred_boxes: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    best_label: Optional[str] = None,
    best_iou: Optional[float] = None,
):
    """
    image: PIL.Image or numpy.ndarray
    gt_box: [x1, y1, x2, y2]
    pred_boxes: [{'bbox_2d': [x1, y1, x2, y2], 'label': str, 'iou': float (optional)}, ...]
    save_path: file path to save; if None, only return the PIL.Image.
    best_label: label of the detection box to highlight.
    best_iou: iou of the best box; used only for the title (optional).
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        vis_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        vis_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    draw = ImageDraw.Draw(vis_img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    if gt_box is not None:
        x1, y1, x2, y2 = map(int, gt_box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        draw.text((x1 + 3, max(0, y1 - 18)), "GT", fill=(0, 255, 0), font=font)

    for item in pred_boxes:
        box = item.get("bbox_2d", None)
        label = item.get("label", "unknown")
        iou = item.get("iou", None)
        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)

        if best_label is not None and label == best_label:
            color = (255, 255, 0)   # yellow
            width = 5
        else:
            color = (255, 0, 0)     # red
            width = 3

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        text = label
        if iou is not None:
            text += f" | IoU={iou:.3f}"

        draw.text((x1 + 3, y1 + 3), text, fill=color, font=font)

    if best_label is not None:
        title = f"best={best_label}"
        if best_iou is not None:
            title += f", IoU={best_iou:.3f}"
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vis_img.save(save_path)

    return vis_img


def extract_question_Guaranteed(text: str):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"</?think>", "", text)

    m = re.search(r'([^?？\n]*[?？])', text)
    return m.group(1).strip() if m else None


def add_talk_before_question(text: str):
    m = re.search(r'([^?？\n]*[?？])', text)
    if not m:
        return text
    start = m.start(1)
    return text[:start] + "<talk>" + text[start:]


def remove_instance_qa(dialogs, instance_id):
    """
    Remove all QA pairs whose navigator thinking mentions table_13:
    - Check whether the navigator's thinking contains table_13.
    - If it does, drop that navigator turn and the oracle turn immediately following it.
    """
    new_dialogs = []
    i = 0

    while i < len(dialogs):
        item = dialogs[i]

        if item.get("role") == "navigator":
            thinking = item.get("thinking", "")

            if instance_id in thinking :
                i += 1
                if i < len(dialogs) and dialogs[i].get("role") == "oracle":
                    i += 1
                continue

        new_dialogs.append(item)
        i += 1

    return new_dialogs


def _extract_position(pos_like):
    if isinstance(pos_like, dict):
        if "agent_state" in pos_like and "position" in pos_like["agent_state"]:
            pos_like = pos_like["agent_state"]["position"]
        elif "position" in pos_like:
            pos_like = pos_like["position"]
        else:
            raise ValueError(f"Unsupported dict position format: {pos_like.keys()}")

    pos = np.asarray(pos_like, dtype=np.float32).reshape(-1)
    if pos.shape[0] < 3:
        raise ValueError(f"Position must have at least 3 dims, got {pos_like}")
    return pos[:3]


def _planar_distance(a, b):
    a = _extract_position(a)
    b = _extract_position(b)
    return float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))


def _get_unique_dialog_turn_ids(dialogs, step_id: Optional[int] = None) -> List[int]:
    turn_ids = []
    for d in dialogs:
        if "true_idx" not in d:
            continue
        t = int(d["true_idx"])
        if step_id is not None and t >= step_id:
            continue
        turn_ids.append(t)
    return sorted(set(turn_ids))


def select_dialog_turn_ids(
    dialogs: List[dict],
    step_id: int,
    path_list: Optional[List[Sequence[float]]] = None,
    mode: Union[str, List[str]] = "all",
    recent_step_n: Optional[int] = None,
    recent_n: Optional[int] = None,
    radius: float = 8.0,
    current_position: Optional[Sequence[float]] = None,
    combine: str = "union",
) -> List[int]:
    """
    Return the list of true_idx values for the dialog turns that should be kept.

    mode:
    - "all"
    - "recent_steps"    : dialogs from the most recent n steps before the current step
    - "recent_dialogs"  : the last n dialogs
    - "spatial_local"   : dialogs that happened within `radius` meters of the current position
    """
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = list(mode)

    turn_ids = _get_unique_dialog_turn_ids(dialogs, step_id=step_id)
    if not turn_ids:
        return []

    selected_sets = []
    for m in modes:
        if m == "all":
            selected_sets.append(set(turn_ids))

        elif m == "recent_steps":
            if recent_step_n is None:
                raise ValueError("recent_steps mode requires recent_step_n")
            start_step = max(0, step_id - recent_step_n)
            selected_sets.append({t for t in turn_ids if start_step <= t < step_id})

        elif m == "recent_dialogs":
            if recent_n is None:
                raise ValueError("recent_dialogs mode requires recent_n")
            if recent_n <= 0:
                selected_sets.append(set())
            else:
                selected_sets.append(set(turn_ids[-recent_n:]))

        elif m == "spatial_local":
            if path_list is None or len(path_list) == 0:
                raise ValueError("spatial_local mode requires non-empty path_list")

            if current_position is None:
                ref_idx = min(max(step_id, 0), len(path_list) - 1)
                current_position = path_list[ref_idx]

            cur_pos = _extract_position(current_position)
            local_ids = set()

            for t in turn_ids:
                if 0 <= t < len(path_list):
                    dialog_pos = path_list[t]
                    if _planar_distance(dialog_pos, cur_pos) <= radius:
                        local_ids.add(t)

            selected_sets.append(local_ids)

        else:
            raise ValueError(f"Unknown dialog selection mode: {m}")

    if combine == "union":
        final_ids = set().union(*selected_sets)
    elif combine == "intersection":
        final_ids = selected_sets[0].copy()
        for s in selected_sets[1:]:
            final_ids &= s
    else:
        raise ValueError(f"Unknown combine mode: {combine}")
    print('[DEBUG]: dialog_turn_ids turn_ids =', turn_ids)
    print('[DEBUG]: select_dialog_turn_ids final_ids =', final_ids)
    return sorted(final_ids)


def filter_dialogs_for_input(
    dialogs: List[dict],
    step_id: int,
    path_list: Optional[List[Sequence[float]]] = None,
    mode: Union[str, List[str]] = "all",
    recent_step_n: Optional[int] = None,
    recent_n: Optional[int] = None,
    radius: float = 8.0,
    current_position: Optional[Sequence[float]] = None,
    combine: str = "union",
):
    dialogs_idx = select_dialog_turn_ids(
        dialogs=dialogs,
        step_id=step_id,
        path_list=path_list,
        mode=mode,
        recent_step_n=recent_step_n,
        recent_n=recent_n,
        radius=radius,
        current_position=current_position,
        combine=combine,
    )
    keep = set(dialogs_idx)
    selected_dialogs = [d for d in dialogs if int(d.get("true_idx", -1)) in keep]
    return selected_dialogs, dialogs_idx


def _extract_uncertain_label_set(think_text, instance_type):
    if not think_text:
        return set()
    tail = think_text.split('Uncertain')[-1]
    labels = extract_typed_instances(tail, instance_type)

    return {lb for lb in set(labels)}


def _extract_frame_label_set(frame_bbox, instance_type):
    prefix = f"{instance_type}_"
    label_set = set()
    for obj in frame_bbox or []:
        lb = obj.get("label")
        if isinstance(lb, str) and lb.startswith(prefix):
            label_set.add(lb)
    return label_set


def _parse_frame_questions_from_talk(talk_text):
    """
    Parse text like:
    Frame_1: Which option matches the chair?
    Frame_3: Which option matches the chair?
    Frame_c: Which option matches the chair?

    Returns:
    {
        0: "Which option matches the chair?",   # Frame_1
        2: "Which option matches the chair?",   # Frame_3
        "current": "Which option matches the chair?"  # Frame_c
    }
    """
    res = {}
    if not talk_text:
        return res

    pattern = r"Frame_([0-9]+|c)\s*:\s*(.*?)(?=\s*Frame_(?:[0-9]+|c)\s*:|$)"
    matches = re.findall(pattern, talk_text, flags=re.DOTALL | re.IGNORECASE)

    for frame_tag, question in matches:
        question = question.strip()
        if not question:
            continue

        if str(frame_tag).lower() == "c":
            res["current"] = question
        else:
            frame_idx = int(frame_tag) - 1  # Frame_1 -> 0
            if frame_idx >= 0:
                res[frame_idx] = question

    return res


def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

"""Evaluator for the IIGN task family (dialog-capable / NPC-aware).

This module bundles ``VLNEvaluator`` and every helper it depends on. The
unified entrypoint ``evaluation/evaluate_navigation.py`` obtains this class
through ``tasks.get_evaluator``.
"""

import os
import copy
import json
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple, Any, Union, Set
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import tqdm
import quaternion
import itertools
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time
import random
import re
import torch
from torch import Tensor
from transformers.image_utils import to_numpy_array
from depth_camera_filtering import filter_depth
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from copy import deepcopy
import shutil
from collections import Counter
import transformers
from transformers import DynamicCache
import habitat
import habitat_sim
from habitat import logger, Env
from habitat_extensions import measures
import habitat_extensions.instance_image_nav
# import habitat_extensions.uncertainty_task
from habitat_extensions.dialog_dataset import DialogDatasetV1
from habitat_extensions.env.objectnav_env import ObjectNavEnv
from habitat_extensions.PSL_dataset import AttrObjNavDatasetV1
from habitat_extensions.uncertainty_dataset import LanguageUncertaintyDatasetV1
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
    append_text_underneath_image
)

from omegaconf import OmegaConf, DictConfig, open_dict
from npc.simple_npc.simple_npc import SimpleNPC
from npc.utils.get_description import get_path_description, get_path_description_without_additional_info
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import rho_theta
from vlfm.obs_transformers.utils import image_resize
from dist import *
from .pair_question import match_utterance

DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
LABEL_PATTERN = re.compile(r"\b[a-zA-Z]+_\d+\b")

TEMPLATES_TEST = [
    {
        "meta": {"scene_type": "single_candidate", "slots": ["instance"]},
        "question": [
            "Is this the {instance} you're looking for?",
            "Is this the target {instance}?",
            "Is this the correct {instance}?",
            "Is this the right {instance}?",
            "Is this the {instance} you need?",
            "Does this look like the {instance} you want?",
            "Is this the {instance} you meant?",
            "Can you confirm this is the {instance}?",
            "Is this the {instance} you asked for?",
            "Is this the {instance} we are searching for?",
            "Is this one the {instance}?",
            # "Is the {instance}"
        ],
    },
    {
        "meta": {"scene_type": "multi_all_unseen", "slots": ["instance", "choice_ordinal"]},
        "question": [
            "Which one is the {instance}?",
            "Do you see the {instance} among these?",
            "Is the {instance} in this set? If so, which one?",
            "Which option matches the {instance}?",
            "Can you point out the {instance} here?",
            "Which candidate is the {instance}?",
            "Which one should I choose for the {instance}?",
            "Is any of these the {instance}? Which one?",
            "Which one looks like the target {instance}?",
            "Which one corresponds to the {instance}?",
            "Among these, which is the {instance}?",
            "Is any of these the {instance}?",
        ],
        "answer": {
            "choice": [
                "The {choice_ordinal} one.",
                "{choice_ordinal}.",
                "Pick the {choice_ordinal} one.",
                "It's the {choice_ordinal} one.",
                "Choose the {choice_ordinal}.",
                "Go to the {choice_ordinal} one.",
                "The {choice_ordinal} option.",
                "Select the {choice_ordinal}.",
                "The {choice_ordinal} candidate.",
                "I mean the {choice_ordinal} one.",
                "The {choice_ordinal} one is correct.",
                "The {choice_ordinal} is the {instance}.",
            ],
            "none": [
                "None of them.",
                "None of these.",
                "No, the {instance} isn't here.",
                "No, none of these match.",
                "Neither-it's not shown here.",
                "No, it's not among these options.",
                "No, none of the candidates fit.",
                "No match here.",
                "No, this set doesn't include the {instance}.",
                "None of these is the target.",
                "No, keep searching.",
                "No, all of them are wrong.",
            ],
        },
    },
    {
        "meta": {
            "scene_type": "multi_confirmed_partial_except",
            "slots": ["instance", "except_clause", "choice_ordinal"]
        },
        "question": {
            "single": [
                "I've ruled out some candidates. Is the {except_clause} the target {instance}?",
                "Only the {except_clause} is still possible in this view. Is it the {instance}?",
                "We can ignore the others. Is the {except_clause} the correct {instance}?",
                "Given what's ruled out, is the {except_clause} the {instance} we want?",
                "After eliminating the rest, is the {except_clause} the target {instance}?",
                "Is the remaining candidate-the {except_clause}-the {instance}?",
                "I narrowed it down to the {except_clause} in this view. Is that the {instance}?",
                "Is the {except_clause} the right {instance}, yes or no?",
                "Can you confirm the {except_clause} is the {instance}?",
                "Final check: is the {except_clause} the {instance}?",
                "Does the {except_clause} match the {instance}?",
                "Is the {except_clause} the target instance?",
            ],
            "double": [
                "I've ruled out some candidates. In the current view, among the {except_clause}, which is the {instance}?",
                "Only the {except_clause} candidates remain uncertain in the current view. Which one is the target {instance}?",
                "I narrowed it down to the {except_clause} in the current view. Which one should I pick for {instance}?",
                "In the current view, between the uncertain options ({except_clause}), which is the {instance}?",
                "In this view, the uncertain candidates are {except_clause}. Which one is the {instance}?",
                "In the current view, I'm down to {except_clause}. Which one matches the {instance}?",
                "In this view, from {except_clause}, which candidate is the {instance}?",
                "In the current view, which of the uncertain candidates ({except_clause}) is the {instance}?",
                "In this view, please choose the {instance} from the uncertain options: {except_clause}.",
                "In the current view, among {except_clause}, which one should I select for {instance}?",
                "In this view, we only have {except_clause} left. Which one is correct for {instance}?",
                "In the current view, pick the {instance} from {except_clause}.",
            ],
        },
        "answer": {
            "single": {
                "positive": [
                    "Yes.",
                    "That's it!",
                    "Correct.",
                    "Yes, this is the one.",
                    "Yes, that's the target.",
                    "Exactly.",
                    "Yep.",
                    "Yes-pick this one.",
                    "Confirmed.",
                    "This is it.",
                    "Yes, this matches.",
                ],
                "negative": [
                    "No.",
                    "Not this one.",
                    "No, that's not it.",
                    "No, keep looking.",
                    "No, wrong {instance}.",
                    "Nope.",
                    "Not the target.",
                    "No, this isn't the {instance}.",
                    "No, skip this one.",
                    "No, try another.",
                    "No, this doesn't match.",
                    "No, not this {instance}.",
                ],
            },
            "double": {
                "choice": [
                    "The {choice_ordinal} one.",
                    "The {choice_ordinal}.",
                    "Pick the {choice_ordinal} one.",
                    "It's the {choice_ordinal} one.",
                    "Choose the {choice_ordinal}.",
                    "Go to the {choice_ordinal} one.",
                    "The {choice_ordinal} option.",
                    "Select the {choice_ordinal}.",
                    "The {choice_ordinal} candidate.",
                    "I mean the {choice_ordinal} one.",
                    "The {choice_ordinal} one is correct.",
                    "The {choice_ordinal} is the {instance}.",
                ],
                "none": [
                    "None of them.",
                    "Neither.",
                    "Neither one.",
                    "No, neither is correct.",
                    "No, none of these matches.",
                    "No, both are wrong.",
                    "Neither candidate is the {instance}.",
                    "No match among the remaining options.",
                    "None of the remaining candidates is correct.",
                    "No, keep looking.",
                    "No, not these.",
                    "No, it's neither of the two.",
                ],
            },
        },
    },
]

                

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import (
    EpisodeMeta,
    IoU,
    _extract_frame_label_set,
    _extract_uncertain_label_set,
    _parse_frame_questions_from_talk,
    add_more_new_instance,
    denorm_bbox,
    draw_bboxes,
    extract_bbox_blocks_from_chat_sources,
    filter_dialogs_for_input,
    format_with_tabs,
    get_bbox_from_mask,
    get_config,
    mask01_to_bbox,
    match_labels_min_frames,
    max_label_id,
    open_image,
    ordinals_clause,
    remove_instance_qa,
    split_and_clean,
    split_llm_output,
    traj_to_actions,
    unify_to_first,
)

def best_iou_for_target_label(bbox_list, seg_map, target_label, scale=1000.0):
    """
    bbox_list: [{'bbox_2d':[...], 'label':...}, ...], typically the bboxes of one frame.
    seg_map:   the GT mask of the current frame, observations['semantic'][0] > 0 or seg_list[idx].
    target_label: the confirmed target label, e.g. 'chair_3'.

    Returns:
        None: cannot be evaluated (e.g. no GT available).
        0.0 : the current frame does not contain the target_label, treated as tracking loss.
        >0  : best IoU between the boxes of this label in the current frame and the GT.
    """
    if target_label is None or bbox_list is None:
        return None

    gt_box = mask01_to_bbox(seg_map)
    target_items = [x for x in bbox_list if x.get("label") == target_label]
    if gt_box is None:
        if len(target_items) == 0:
            return None
        else:
            return 0.0

    
    if len(target_items) == 0:
        return 0.0

    pred_items = denorm_bbox(target_items, seg_map, scale=scale, clamp=True)
    return max(IoU(item["bbox_2d"], gt_box) for item in pred_items)

def classify_episode_failure(success, ever_observed, ever_grounded,
                             tracking_iou_history, distance_to_goal,
                             track_iou_thr=0.2, neighborhood_thr=1.0):
    """
    Failure taxonomy:
    1) Never observed -> Exploration Error
    2) Observed but not grounded -> Grounding Error
    3) Grounded, but the last two tracking IoUs are not both > 0.2 -> Tracking Error
    4) Tracking OK, but did not reach the target neighborhood -> Navigation Error
    5) Reached the neighborhood but failed to stop -> Stop Error
    6) success -> Success
    """
    
    if success:
        return "Success", True, tracking_iou_history[-2:]
    
    if not ever_observed:
        return "Exploration Error", False, tracking_iou_history[-2:]

    if not ever_grounded:
        return "Grounding Error", False, tracking_iou_history[-2:]
    last_five = tracking_iou_history[-5:]
    low_iou_cnt = sum(iou < 0.15 for iou in last_five)
    tracking_ok = low_iou_cnt < 3

    if not tracking_ok:
        return "Tracking Error", False, last_five

    if distance_to_goal > neighborhood_thr:
        return "Navigation Error", True, last_five

    return "Stop Error", True, last_five

def merge_tracking_and_current_outputs(
    tracking_output,
    llm_outputs,
    instance_type,
    messages,
    num_tracking_frames: int = 0,
):
    """
    Merge tracking-frame results from tracking_output with the current-frame results from
    llm_outputs. The current frame is always the last frame, padded with [] if it has no bbox.

    Conventions:
    - tracking_output does not contain any question.
    - All questions come from the <talk> section of llm_outputs.
    - Frame_1 / Frame_2 / ... correspond to tracking frames.
    - Frame_c corresponds to the current frame (the last frame).
    - num_tracking_frames: the number of tracking images actually fed to the model, used to
      guarantee len(bbox_blocks) == num_tracking_frames + 1 (including the current frame).

    Extra logic:
    - If instance_type does not match the prefix of the label produced by the model, prefer the
      prefix from the model output (e.g. if the model emits chair_0, use "chair").
    """

    def _infer_effective_instance_type(
        tracking_bbox_blocks,
        current_bbox,
        tracking_output,
        llm_outputs,
        fallback_instance_type,
    ):
        """
        Infer the true instance_type primarily from the labels produced by the model.
        Rules:
        1. First check labels inside bboxes (most reliable).
        2. Otherwise, search the raw text for patterns like xxx_0 / xxx_1.
        3. If neither works, fall back to the given instance_type.
        """

        def _count_prefixes_from_bbox_blocks(bbox_blocks):
            counter = Counter()
            for block in bbox_blocks:
                for obj in block or []:
                    lb = obj.get("label")
                    if not isinstance(lb, str):
                        continue
                    lb = lb.strip()
                    m = re.match(r"^(.*)_(\d+)$", lb)
                    if m:
                        counter[m.group(1)] += 1
            return counter

        def _count_prefixes_from_text(text):
            counter = Counter()
            if not text:
                return counter
            pattern = r"\b([A-Za-z][A-Za-z0-9-]*(?:_[A-Za-z0-9-]+)*)_(\d+)\b"
            for m in re.finditer(pattern, text):
                counter[m.group(1)] += 1
            return counter

        bbox_counter = Counter()
        bbox_counter.update(_count_prefixes_from_bbox_blocks(tracking_bbox_blocks))
        bbox_counter.update(_count_prefixes_from_bbox_blocks([current_bbox]))
        if bbox_counter:
            return bbox_counter.most_common(1)[0][0]

        text_counter = Counter()
        text_counter.update(_count_prefixes_from_text(tracking_output))
        text_counter.update(_count_prefixes_from_text(llm_outputs))
        if text_counter:
            return text_counter.most_common(1)[0][0]

        return fallback_instance_type

    tracking_bbox_blocks = extract_bbox_blocks_from_chat_sources(tracking_output) or []
    current_bbox_blocks = extract_bbox_blocks_from_chat_sources(llm_outputs) or []

    current_bbox = current_bbox_blocks[-1] if len(current_bbox_blocks) > 0 else []

    if num_tracking_frames > 0:
        n = len(tracking_bbox_blocks)
        if n < num_tracking_frames:
            tracking_bbox_blocks = [[] for _ in range(num_tracking_frames - n)] + tracking_bbox_blocks
        elif n > num_tracking_frames:
            tracking_bbox_blocks = tracking_bbox_blocks[-num_tracking_frames:]

    bbox_blocks = tracking_bbox_blocks + [current_bbox]

    tracking_think, _ = split_llm_output(tracking_output)
    current_think, current_question = split_llm_output(llm_outputs)

    effective_instance_type = _infer_effective_instance_type(
        tracking_bbox_blocks=tracking_bbox_blocks,
        current_bbox=current_bbox,
        tracking_output=tracking_output,
        llm_outputs=llm_outputs,
        fallback_instance_type=instance_type,
    )

    q_list = [None] * len(bbox_blocks)
    frame_question_map = _parse_frame_questions_from_talk(current_question)

    for k, q in frame_question_map.items():
        if k == "current":
            continue
        if 0 <= k < len(tracking_bbox_blocks):
            q_list[k] = q

    if "current" in frame_question_map:
        q_list[-1] = frame_question_map["current"]

    label_set = _extract_uncertain_label_set(tracking_think, effective_instance_type)
    label_set |= _extract_uncertain_label_set(current_think, effective_instance_type)
    label_set |= _extract_frame_label_set(current_bbox, effective_instance_type)

    label_set = add_more_new_instance(label_set, tracking_output, messages, effective_instance_type)
    label_set = add_more_new_instance(label_set, llm_outputs, messages, effective_instance_type)

    return bbox_blocks, q_list, label_set

def merge_tracking_and_current_outputs_(
    tracking_output,
    llm_outputs,
    instance_type,
    messages,
    num_tracking_frames: int = 0,
):
    """
    Merge tracking-frame results from tracking_output with the current-frame results from
    llm_outputs. The current frame is always the last frame, padded with [] if it has no bbox.

    Conventions:
    - tracking_output does not contain any question.
    - All questions come from the <talk> section of llm_outputs.
    - Frame_1 / Frame_2 / ... correspond to tracking frames.
    - Frame_c corresponds to the current frame (the last frame).
    - num_tracking_frames: the number of tracking images actually fed to the model, used to
      guarantee len(bbox_blocks) == num_tracking_frames + 1 (including the current frame).
    """

    tracking_bbox_blocks = extract_bbox_blocks_from_chat_sources(tracking_output) or []
    current_bbox_blocks = extract_bbox_blocks_from_chat_sources(llm_outputs) or []

    current_bbox = current_bbox_blocks[-1] if len(current_bbox_blocks) > 0 else []

    if num_tracking_frames > 0:
        n = len(tracking_bbox_blocks)
        if n < num_tracking_frames:
            tracking_bbox_blocks = [[] for _ in range(num_tracking_frames - n)] + tracking_bbox_blocks
        elif n > num_tracking_frames:
            tracking_bbox_blocks = tracking_bbox_blocks[-num_tracking_frames:]

    bbox_blocks = tracking_bbox_blocks + [current_bbox]

    tracking_think, _ = split_llm_output(tracking_output)
    current_think, current_question = split_llm_output(llm_outputs)

    q_list = [None] * len(bbox_blocks)
    frame_question_map = _parse_frame_questions_from_talk(current_question)

    for k, q in frame_question_map.items():
        if k == "current":
            continue
        if 0 <= k < len(tracking_bbox_blocks):
            q_list[k] = q

    if "current" in frame_question_map:
        q_list[-1] = frame_question_map["current"]

    label_set = _extract_uncertain_label_set(tracking_think, instance_type)
    label_set |= _extract_uncertain_label_set(current_think, instance_type)
    label_set |= _extract_frame_label_set(current_bbox, instance_type)

    label_set = add_more_new_instance(label_set, tracking_output, messages, instance_type)
    label_set = add_more_new_instance(label_set, llm_outputs, messages, instance_type)

    return bbox_blocks, q_list, label_set

class VLNEvaluator:
    def __init__(
        self,
        npc: SimpleNPC,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 1,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
        model_2: Any = None,
    ):
        self.npc = npc
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        # self.config = get_habitat_config(config_path)
        self.config = get_config(config_path, args.baseline_config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.success_distance = self.config.habitat.task.measurements.success.success_distance
        self.different_expnav = False
        # history frame mode
        self.dialog_history_mode = args.dialog_history_mode
        self.dialog_recent_step_n = args.dialog_recent_step_n
        self.dialog_recent_n = args.dialog_recent_n
        self.dialog_radius = args.dialog_radius
        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.exp.task = args.task
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        
        self.model = model
        if model_2 is not None:
            self.model_2 = model_2
        self.processor = processor
        self.W = 640
        self.H = 480
        # pointnav policy
        # self._pointnav_policy = WrappedPointNavResNetPolicy('data/pointnav_weights.pth')
        # self._pointnav_depth_image_shape = (224, 224)
        # self._pointnav_stop_radius = 0.5

        # pixelnav policy
        # from pixel_nav.policy_agent import Policy_Agent
        # self.pixel_nav_agent = Policy_Agent(
        #     model_path="/mnt/hwfile/OpenRobotLab/wanchenyang/share/pixel_nav/pixel_nav_policy_2025-04-21 15:28:49/pixelnav_50.ckpt",
        #     device=self.device,
        #     max_token_length=20,
        # )

        # --- Prompt templates aligned with training (data_qwen_dialog_with_onestep_v3.py) ---
        # Turn 1 Human: instruction + history + (pre_lookdown) + tracking_instruction + decision_instruction
        # The template uses placeholders: <instruction>, <history>, <tracking_instruction>
        # Two variants: with and without pre_lookdown image
        prompt_with_lookdown = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "These are your historical observations: <history>. "
            "Your previous look down image is <image>. "
            "<tracking_instruction>"
        )
        prompt_without_lookdown = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "These are your historical observations: <history>. "
            "<tracking_instruction>"
        )
        answer = ""
        self.conversation_with_lookdown = [{"from": "human", "value": prompt_with_lookdown}, {"from": "gpt", "value": answer}]
        self.conversation_without_lookdown = [{"from": "human", "value": prompt_without_lookdown}, {"from": "gpt", "value": answer}]
        # Keep legacy references for non-dialog / objectnav paths
        prompt_nav = 'You are an autonomous navigation assistant. Your task is to <instruction>. You need to navigate to the target. When you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees then output the next point coordinates in the image. In case the point is out of view, utilize the turn actions: TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. Please output STOP when you have successfully completed the task.'
        self.conversation_nav = [{"from": "human", "value": prompt_nav}, {"from": "gpt", "value": answer}]
        # Decision instruction (matching training lines 1066-1076)
        # self.decision_instruction = (
        #     "Your current image is <image>. Please first output the tracking result in this image, "
        #     "and then decide the next action based on all the context. "
        #     "You can either choose to move or talk. "
        #     "If you choose to talk, you may ask a question about either a tracking image or the current image: "
        #     "use 'Frame_1: ...? Frame_2: ...?' for tracking images, and use 'Frame_c: ...?' for the current image. "
        #     "If choosing to move, when you want to output a waypoint, you need to TILT DOWN (↓) by 30 degrees "
        #     "and then output the next point coordinates in the image. "
        #     "If the point is out of view, use the turn actions TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. "
        #     "Please output STOP when you have successfully completed the task. "
        # )
        self.decision_instruction = (
            "Your current image is <image>. "
            "Decide the next action based on all the context. "
            "You can either choose to move or talk. "
            "If you choose to talk, you may only ask a question about one of the recent tracking images: "
            "use 'Frame_1: ...? Frame_2: ...?' for tracking images. "
            "If choosing to move, when you want to output a waypoint, you need to TILT DOWN (↓) by 30 degrees "
            "and then output the next point coordinates in the image. "
            "If the point is out of view, use the turn actions TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. "
            "Please output STOP when you have successfully completed the task. "
        )
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        
        self.will_see_conjunctions = [
            'I will see the next waypoint.',
            'I will spot the next goal ahead.',
            'The next destination will come into view.',
            'The next landmark will become visible.',
            'I will see the next target clearly.',
            'The next waypoint will be within my sight.',
            'I will spot the next goal in the distance.',
            'I will see the next destination.',
            'The next target will come into focus.',
            'I will see the next navigation point.'
        ]
        
        self.see_conjunctions = [
            'I can see the next waypoint.',
            'I can spot the next goal ahead.',
            'The next destination is visible.',
            'The next landmark is in view.',
            'I can see the next target clearly.',
            'The next waypoint is within my field of view.',
            'I can spot the next goal in the distance.',
            'I can see the next destination.',
            'The next target is in focus.',
            'I can see the next navigation point.'
        ]

        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3],
            "↓": [5],
        })
                                
        self.objectnav_instructions =[
            # 'Go to find the {target_object}, move close to the object, and stop.',
            # 'Search for the {target_object}, move close to the object, and stop.',
            # 'Find the {target_object} in the scene, move close to the object, and stop.',
            # 'Explore the area until you find the {target_object}, move close to the object, and stop.',
            # 'Move towards the {target_object}, move close to the object, and stop.',
            # 'Navigate to the {target_object}, move close to the object, and stop.',
            # 'Find and stop in front of the {target_object}.',
            # 'Move close to the area where the {target_object} is located, and stop.'
            "search for {target_object}."
        ]
        
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        self.max_episode_steps = 500
    
    def make_fake_dialog(self, segments, dialogs, instance_type, step_id, scene_id, episode_id):
        history_dialogs = list(dialogs)

        bbox_list = []
        max_id = max_label_id(history_dialogs, instance_type)
        next_id = max_id + 1
        for seg in segments:
            if seg.sum() > 100:
                bbox = mask01_to_bbox(seg)
                bbox = [
                    int(round(bbox[0] * 1000 / self.W)),
                    int(round(bbox[1] * 1000 / self.H)),
                    int(round(bbox[2] * 1000 / self.W)),
                    int(round(bbox[3] * 1000 / self.H)),
                ]
                bbox_list.append({"bbox_2d": bbox, "label": f"{instance_type}_{next_id}"})
                next_id += 1

        cur_num = len(bbox_list)
        first_obj = bbox_list[0]
        bbox_list = sorted(bbox_list, key=lambda x: x["bbox_2d"][0])
        gt_idx = bbox_list.index(first_obj)
        choice_ordinal = ordinals_clause([gt_idx])

        bbox_message = format_with_tabs(bbox_list)

        if cur_num == 1:
            question = f"Is this the {instance_type} you asked for?"
            answer = "Yes."
        else:
            question = f"Do you see the {instance_type} among these?"
            answer = f"The {choice_ordinal} one."

        dialogs[:] = [d for d in dialogs if d.get("true_idx") != step_id]
        dialogs.append({
            "role": "navigator",
            "message": question,
            "true_idx": step_id,
            "thinking": (
                f"<think>I am looking for a specific {instance_type}. "
                f"In this image, I detected {cur_num} {instance_type}(s).\n"
                f"{bbox_message}"
                f" 0 of them have been checked and are not the target instance."
                f" The remaining {cur_num} are still uncertain.</think>"
            )
        })
        dialogs.append({"role": "oracle", "message": answer, "true_idx": step_id})
        self._append_action_log(
            scene_id,
            episode_id,
            [dialogs[-2]["thinking"], dialogs[-2]["message"], dialogs[-1]["message"]],
            step_id=step_id,
        )
        return dialogs, bbox_list

    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        width, height = depth_image.size
        target_h = self.image_processor.crop_size["height"]
        target_w = self.image_processor.crop_size["width"]

        # Keep aspect ratio, then center-crop to processor target size.
        if width < height:
            new_w = target_h
            new_h = int(height * (target_h / width))
        else:
            new_h = target_h
            new_w = int(width * (target_h / height))

        resized = depth_image.resize((new_w, new_h), Image.NEAREST)
        left = (new_w - target_w) / 2
        top = (new_h - target_h) / 2
        right = left + target_w
        bottom = top + target_h
        cropped = resized.crop((left, top, right, bottom))

        img = to_numpy_array(cropped)
        if do_depth_scale:
            img = img / depth_scale
        img = img * (self._max_depth - self._min_depth) + self._min_depth
        return img, (new_w, new_h)
    
    def preprocess_depth_image_v2(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size["height"]
        target_width = self.image_processor.crop_size["width"]
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale

        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float32)
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = np.array(intrinsic, copy=True)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return transformation_matrix

    def xyz_pitch_to_tf_matrix(self, xyz: np.ndarray, pitch: float) -> np.ndarray:
        """Converts a given position and pitch angle to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """

        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), x],
                [0, 1, 0, y],
                [-np.sin(pitch), 0, np.cos(pitch), z],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return transformation_matrix
    
    def xyz_yaw_pitch_to_tf_matrix(self, xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        rot1 = self.xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        rot2 = self.xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix[:3, :3] = rot1 @ rot2
        transformation_matrix[:3, 3] = xyz
        return transformation_matrix
    
    def pixel_to_gps(self, pixel, depth, intrinsic, tf_camera_to_episodic):
        '''
        Args:
            pixel: (2,) - [u, v] pixel coordinates
            depth: (H, W) - depth image where depth[v, u] gives depth in meters
            intrinsic: (4, 4) - camera intrinsic matrix
            tf_camera_to_episodic: (4, 4) - transformation from camera to episodic frame
        Returns:
            (x, y): (x, y) coordinates in the episodic frame
        '''
        v, u = int(pixel[0]), int(pixel[1])
        v = np.clip(v, 0, depth.shape[0] - 1)
        u = np.clip(u, 0, depth.shape[1] - 1)
        z = float(depth[v, u])
        if z <= 0:
            raise ValueError(f"Invalid depth at pixel {pixel}: {z}")

        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        point_camera = np.array([x, y, z, 1.0], dtype=np.float32)

        # Transform to episodic frame
        point_episodic = tf_camera_to_episodic @ point_camera
        point_episodic = point_episodic[:3] / max(float(point_episodic[3]), 1e-6)
        return float(point_episodic[0]), float(point_episodic[1])  # same as habitat gps
    
    def config_env(self) -> Env:
        env = ObjectNavEnv(config=self.config.habitat, exp_config=self.config.exp)
        # env.sim.recompute_navmesh()
        # env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env
    
    def dot_matrix_two_dimensional(self, image_or_image_path, save_path = None, dots_size_w = 8, dots_size_h = 8, save_img = False, font_path = '/mnt/hwfile/OpenRobotLab/weimeng/fonts/arial.ttf', pixel_goal = None):
        """
        takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
        control args:
        1. dots_size_w: the number of columns of the dots matrix
        2. dots_size_h: the number of rows of the dots matrix
        """
        with open_image(image_or_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            draw = ImageDraw.Draw(img, 'RGB')

            width, height = img.size
            grid_size_w = dots_size_w + 1
            grid_size_h = dots_size_h + 1
            cell_width = width / grid_size_w
            cell_height = height / grid_size_h

            font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40

            target_i = target_j = None
            if pixel_goal is not None:
                y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                # Validate pixel coordinates
                if not (0 <= x_pixel < width and 0 <= y_pixel < height):
                    raise ValueError(f"pixel_goal {pixel_goal} exceeds image dimensions ({width}x{height})")
                
                # Convert to grid coordinates
                target_i = round(x_pixel / cell_width)
                target_j = round(y_pixel / cell_height)
                
                # Validate grid bounds
                if not (1 <= target_i <= dots_size_w and 1 <= target_j <= dots_size_h):
                    raise ValueError(
                        f"pixel_goal {pixel_goal} maps to grid ({target_j},{target_i}), "
                        f"valid range is (1,1)-({dots_size_h},{dots_size_w})"
                    )

            count = 0
            for j in range(1, grid_size_h):
                for i in range(1, grid_size_w):
                    x = int(i * cell_width)
                    y = int(j * cell_height)

                    pixel_color = img.getpixel((x, y))
                    # choose a more contrasting color from black and white
                    if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                        opposite_color = (0,0,0)
                    else:
                        opposite_color = (255,255,255)
                    
                    if pixel_goal is not None and i == target_i and j == target_j:
                        opposite_color = (255, 0, 0)  # Red for target

                    circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                    draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)], fill=opposite_color)

                    text_x, text_y = x + 3, y
                    count_w = count // dots_size_w
                    count_h = count % dots_size_w
                    label_str = f"({count_w+1},{count_h+1})"
                    draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                    count += 1
            if save_img:
                print(">>> dots overlaid image processed, stored in", save_path)
                img.save(save_path)
            return img
    
    def _pointnav(self, goal: np.ndarray, depth: np.ndarray, step_id: int, robot_xy: np.ndarray, robot_heading: float, stop: bool = False) -> Tensor:
        '''
        Args:
            goal (np.ndarray): goal position
            stop (bool): whether to stop
        Returns:
            action: action tensor
        '''

        masks = torch.tensor([step_id != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                print('Pointnav policy reset!')
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        rho, theta = rho_theta(robot_xy, robot_heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                depth,
                (self._pointnav_depth_image_shape[0], self._pointnav_depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        # print(f"_pointnav_stop_radius: {self._pointnav_stop_radius}")
        # print(f"depth for pointnav: {obs_pointnav['depth']}")
        if rho < self._pointnav_stop_radius and stop:
            return 0
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def denorm_pixel(self, coords, scale=1000):
        w_ = int(coords[0]/scale*self.W)
        h_ = int(coords[1]/scale*self.H)
        return [w_, h_]

    def _to_scalar_action(self, action: Any) -> int:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if hasattr(action, "__len__"):
            action = action[0]
        return int(action)

    def _build_episode_meta(self, episode: Any, scene_id: str) -> EpisodeMeta:
        episode_id = int(episode.episode_id)
        return EpisodeMeta(
            scene_id=scene_id,
            episode_id=episode_id,
            episode_instruction=self._build_episode_instruction(episode),
            instance_type=self._extract_instance_type(episode, episode_id),
        )

    def _action_log_path(self, scene_id: str, episode_id: int) -> str:
        return os.path.join(self.output_path, "action", scene_id, f"{episode_id}.txt")

    def _append_action_log(self, scene_id: str, episode_id: int, lines: List[str], step_id: Optional[int] = None) -> None:
        path = self._action_log_path(scene_id, episode_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                if step_id is None:
                    f.write(f"{line}\n")
                else:
                    f.write(f"{step_id} {line}\n")

    def _result_jsonl_path(self) -> str:
        return os.path.join(self.output_path, "result.json")

    def _append_result(self, result: Dict[str, Any]) -> None:
        with open(self._result_jsonl_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _group_episodes_by_scene(self, env: Env) -> Dict[str, List[Any]]:
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        return scene_episode_dict

    def _build_episode_instruction(self, episode: Any) -> str:
        if "coin" in self.config_path:
            return episode.object_category + " | " + episode.instruction
        if "objectnav" in self.config_path:
            return episode.object_category
        if "psl" in self.config_path:
            return episode.object_category
        return episode.instruction.instruction_text

    def _extract_instance_type(self, episode: Any, episode_id: int) -> str:
        if "objectnav" in self.config_path or "psl" in self.config_path or "coin" in self.config_path:
            return episode.object_category
        instance_type = episode.instruction.instruction_text.split(" the ")[-1][:-1]
        # Keep original special-case behavior.
        if episode_id in [485, 487]:
            instance_type = "microwave"
        return instance_type

    def _load_completed_results(
        self,
    ) -> Tuple[Set[Tuple[str, int, str]], List[float], List[float], List[float], List[float]]:
        done_res: Set[Tuple[str, int, str]] = set()
        sucs: List[float] = []
        spls: List[float] = []
        oss: List[float] = []
        nes: List[float] = []
        result_path = self._result_jsonl_path()
        if not os.path.exists(result_path):
            return done_res, sucs, spls, oss, nes

        with open(result_path, "r") as f:
            for line in f.readlines():
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "scene_id" not in res or "episode_id" not in res or "episode_instruction" not in res:
                    continue
                done_res.add((res["scene_id"], int(res["episode_id"]), res["episode_instruction"]))
                sucs.append(res.get("success", 0))
                spls.append(res.get("spl", 0))
                oss.append(res.get("os", 0))
                nes.append(res.get("ne", 0))
        return done_res, sucs, spls, oss, nes

    def eval_action(self, idx) -> None:
        self.model.eval()
        env = self.config_env()
        scene_episode_dict = self._group_episodes_by_scene(env)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        done_res, sucs, spls, oss, nes = self._load_completed_results()
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            
            print(f"scene_id = {scene_id}")
            # episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                # if "en" not in episode.instruction.language:
                #     continue
                meta = self._build_episode_meta(episode, scene_id)
                episode_instruction = meta.episode_instruction
                print("episode start", episode_instruction)
                # ref_path = [episode.start_position] + episode.reference_path + [episode.goals[0].position]
                episode_id = meta.episode_id
                episode_input_tokens_total = 0
                episode_input_tokens_per_step = []
                if (scene_id, episode_id, episode_instruction) in done_res:
                    continue
                env.current_episode = episode
                observations = env.reset()
                instance_type = meta.instance_type
                # observations = {
                #     'rgb': observations.rgb,
                #     'depth': observations.depth,
                #     'compass': observations.compass,
                #     'gps': observations.gps,
                #     'semantic': observations.semantic
                # }
                agent_state = env.sim.get_agent_state()
                rotation = agent_state.rotation
                translation = agent_state.position
                rotation_matrix = quaternion.as_rotation_matrix(rotation)
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation
                    
                agent = ShortestPathFollower(
                                env.sim, 0.25, False
                            )
                self._last_goal = np.zeros(2)

                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
                os.makedirs(os.path.join(self.output_path, 'action', f'{scene_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                raw_rgb_list = []
                action_list = []
                path_list = []
                action_seq = []
                seg_list = []
                track_list = []
                output_ids = None
                
                goal = None
                action = None
                prev_look_image = None
                look_down_image = None
                messages = []
                dialogs_idx = []
                dialogs = []
                # nav_dialogs = []
                allowed_turns = self.args.turn
                agent_found_target = False
                found_target = False
                found_step = -1
                exp2nav_step = -1
                pre_decision_step = -1
                tracking_iou_history = []
                tracking_step_history = []
                step_id2track = {}
                failure_reason = None
                bbox_blocks = None
                # check=False
                confirmed_target_label = None
                while not env.episode_over and (step_id - action_list.count(5)) <= self.max_episode_steps:
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    # tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw) @ self.get_axis_align_matrix()
                    tf_camera_to_episodic = self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ self.get_axis_align_matrix()
                    
                    image = Image.fromarray(rgb).convert('RGB')  # raw observation image 
                    image_size = image.size  #640*480
                    save_raw_image = image.copy()
                    
                    save_dot = False
                    if action == 5:
                        
                        look_down_image = image
                        save_raw_image = look_down_image.copy()
                        
                        if self.args.use_prompt_image:
                            look_down_image = look_down_image.resize((self.args.resize_w, self.args.resize_h))
                            look_down_image = self.dot_matrix_two_dimensional(look_down_image, save_img=False, save_path=f'test_{step_id}.jpg')
                            save_dot = True
                        # rgb_list.append(look_down_image)
                    elif action == 6:
                        pass
                    else:
                        raw_rgb_list.append(image)
                        image = image.resize((self.args.resize_w, self.args.resize_h))
                        rgb_list.append(image)
                        segment = observations["semantic"][0] > 0
                        seg_list.append(segment)
                        min_x, min_y, max_x, max_y = get_bbox_from_mask(segment)
                        if observations["semantic"][0].sum() > 1000 and not (max_x - min_x < 20 or max_y - min_y < 20) and not found_target:
                            found_target = True
                            found_step = int(step_id)
                            # dialogs, fake_bboxes = self.make_fake_dialog(observations["semantic"], dialogs, instance_type, step_id, scene_id, episode_id)
                            # bbox_image, gt_box = draw_bboxes(image, fake_bboxes, is_norm=True, gt_bbox=None)
                            # out_image_path = os.path.join(self.output_path, f"vis_0", scene_id, f"{episode_id:04d}", f"grounding_{step_id:04d}.png")
                            # p.parent.mkdir(parents=True, exist_ok=True)
                            # bbox_image.save(out_image_path)
                            
                    
                    info = env.get_metrics()
                    # if step_id == 270:

                    if len(action_seq) == 0 and goal is None:
                        if action != 5 and action !=6:
                            # Separate tracking frames (exclude current) and current frame
                            cur_segs = seg_list[pre_decision_step+1:]
                            tracking_images = raw_rgb_list[pre_decision_step+1 : len(raw_rgb_list)-1]  # tracking only, exclude current
                            current_image = raw_rgb_list[-1]  # current frame (the latest appended)
                            tracking_offset = 0
                            if len(tracking_images) > 7:
                                tracking_offset = len(tracking_images) - 7
                                tracking_images = tracking_images[-7:]
                                cur_segs = cur_segs[-8:]

                            if agent_found_target and self.different_expnav:
                                sources = copy.deepcopy(self.conversation_nav)
                                if 'objectnav' in self.config_path:
                                    sources[0]["value"] = sources[0]["value"].replace('<instruction>.', random.choice(self.objectnav_instructions).format(target_object=episode.object_category.replace('_', ' ')))
                                # elif 'coin' in self.config_path:
                                #     sources[0]["value"] = sources[0]["value"].replace('<instruction>.', random.choice(self.objectnav_instructions).format(target_object=episode.object_category.replace('_', ' '))+" "+episode.instruction)
                                else:
                                    sources[0]["value"] = sources[0]["value"].replace('<instruction>', 'search for the <category>').replace('<category>', instance_type)
                                # nav mode: keep legacy behavior, no tracking/decision restructuring
                                input_images = [current_image]
                                history_id = []
                            else:
                                # --- Build conversation matching training format ---
                                # Build instruction text
                                if 'objectnav' in self.config_path:
                                    instruction_text = random.choice(self.objectnav_instructions).format(target_object=episode.object_category.replace('_', ' '))
                                # elif 'coin' in self.config_path:
                                #     instruction_text = random.choice(self.objectnav_instructions).format(target_object=episode.object_category.replace('_', ' '))+" "+episode.instruction
                                else:
                                    instruction_text = f'search for the {instance_type}'

                                if step_id == 0:
                                    history_id = []
                                    dialogs_idx = []
                                    placeholder = ''
                                else:
                                    if agent_found_target and self.different_expnav:
                                        dialogs_idx = np.sort(list(set([i['true_idx'] for i in dialogs[-2:]]))).tolist()
                                        history_id = np.unique(np.linspace(exp2nav_step, step_id - 1, self.num_history, dtype=np.int32)).tolist()
                                    else:         
                                        selected_dialogs, dialogs_idx = filter_dialogs_for_input(
                                            dialogs=dialogs,
                                            step_id=step_id,
                                            path_list=path_list,
                                            mode=self.dialog_history_mode,
                                            recent_step_n=getattr(self, "dialog_recent_step_n", None),
                                            recent_n=getattr(self, "dialog_recent_n", None),
                                            radius=getattr(self, "dialog_radius", 8.0),
                                            current_position=path_list[min(step_id, len(path_list)-1)] if len(path_list) > 0 else None,
                                        )
                                        if agent_found_target and dialogs:
                                            last_dialogs = dialogs[-2:]
                                            existing_indices = set(d['true_idx'] for d in selected_dialogs)
                                            for d in last_dialogs:
                                                if d['true_idx'] not in existing_indices:
                                                    selected_dialogs.append(d)
                                        dialogs_idx = np.sort(list(set([i['true_idx'] for i in selected_dialogs]))).tolist()
                                        history_id = np.unique(np.linspace(0, pre_decision_step, self.num_history, dtype=np.int32)).tolist()

                                    history_id = np.sort(
                                        np.unique(
                                            np.concatenate([
                                                history_id,
                                                dialogs_idx
                                            ]).astype(np.int32)
                                        )
                                    ).tolist()
                                    # Build history dialog placeholder (same format as training build_dialog_history)
                                    placeholder = [''] * (len(history_id)+1)
                                    for n in dialogs_idx:
                                        pos = history_id.index(n)
                                        output = ""
                                        for dialog in dialogs:
                                            if dialog['true_idx'] == n:
                                                thinking = dialog.get('thinking', "")
                                                output += thinking
                                                if dialog['role'] == 'navigator':
                                                    output += f"<dialog><|{dialog['role']}|>{dialog['message']}"
                                                elif dialog['role'] == 'oracle':
                                                    output += f"<|{dialog['role']}|>{dialog['message']}</dialog>"
                                                else:
                                                    raise ValueError(f"unknown role: {dialog.get('role')!r}")
                                        placeholder[pos+1] = output
                                    placeholder = (DEFAULT_IMAGE_TOKEN + '\n').join(placeholder)

                                # Build tracking instruction (matching training lines 1056-1064)
                                tracking_instruction = ""
                                if tracking_images:
                                    tracking_imgs_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * len(tracking_images)
                                    tracking_instruction = (
                                        "Now, first track object identities in the recent observations before the current decision: "
                                        f"{tracking_imgs_tokens}. "
                                        "These tracking images are only for maintaining consistent identities, matching previously seen instances, "
                                        "and summarizing newly discovered instances."
                                        "Do not decide the final action yet."
                                    )

                                # Select conversation template (with or without pre_lookdown)
                                if prev_look_image is not None:
                                    sources = copy.deepcopy(self.conversation_with_lookdown)
                                else:
                                    sources = copy.deepcopy(self.conversation_without_lookdown)

                                # Fill in template placeholders
                                sources[0]["value"] = sources[0]["value"].replace('<instruction>', instruction_text)
                                sources[0]["value"] = sources[0]["value"].replace('<history>', placeholder)
                                sources[0]["value"] = sources[0]["value"].replace('<tracking_instruction>', tracking_instruction)

                                # NOTE: decision_instruction is NOT appended here.
                                # It will be added as Turn 2 Human after model generates tracking result.

                                history_id = sorted(history_id)
                                print('history_id', step_id, history_id)

                                # Build Turn 1 input_images (history -> pre_lookdown -> tracking, NO current)
                                input_images = [rgb_list[i] for i in history_id]
                                if prev_look_image is not None:
                                    input_images.append(prev_look_image)
                                input_images.extend(tracking_images)

                            input_img_id = 0
                            
                        elif action == 6:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            messages.append({
                                            'role': 'user',
                                            'content': [
                                                {
                                                    'type': 'text',
                                                    'text': npc_answers
                                                }
                                            ]
                                        })
                            # Inference for action==6 (oracle reply -> move decision)
                            inputs = self.processor.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors="pt",
                            ).to(self.model.device)
                            step_input_tokens = int(inputs["attention_mask"][0].sum().item())
                            episode_input_tokens_total += step_input_tokens
                            # episode_input_tokens_per_step.append(step_input_tokens)
                            with torch.no_grad():
                                output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
                            llm_outputs = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            messages.append({
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': llm_outputs}]
                            })
                            print('step_id:', step_id, 'action==6 output text:', llm_outputs)

                        else:
                            assert action == 5 # last action is look down
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            input_images += [look_down_image]
                            input_img_id = -1
                            # action==5: lookdown prompt as user message
                            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                            prompt += '. Locate the goal location on the floor in this image'
                            sources[0]["value"] += f" {prompt}."
                            prompt_instruction = copy.deepcopy(sources[0]["value"])
                            parts = split_and_clean(prompt_instruction)

                            content = []
                            for i in range(len(parts)):
                                if parts[i] == "<image>":
                                    content.append({"type": "image", "image": input_images[input_img_id]})
                                    input_img_id += 1
                                else:
                                    content.append({"type": "text", "text": parts[i]})
                            messages.append({'role': 'user', 'content': content})

                            # Inference for action==5 (lookdown -> pixel coordinates)
                            inputs = self.processor.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors="pt",
                            ).to(self.model.device)
                            step_input_tokens = int(inputs["attention_mask"][0].sum().item())
                            episode_input_tokens_total += step_input_tokens
                            # episode_input_tokens_per_step.append(step_input_tokens)
                            with torch.no_grad():
                                output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9)
                            llm_outputs = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            # Retry with do_sample=True if output contains '<move>↓'
                            _retry_count = 0
                            while '<move>↓' in llm_outputs and _retry_count < 5:
                                _retry_count += 1
                                print(f'[Retry {_retry_count}] llm_outputs contains <move>↓, re-sampling...')
                                with torch.no_grad():
                                    output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9)
                                llm_outputs = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            print('step_id:', step_id, 'action==5 output text:', llm_outputs)

                        if action != 6 and action != 5:
                            # ===== Two-step inference (aligned with training multi-turn format) =====
                            # --- Turn 1: Build user message for tracking (history + tracking images, NO current) ---
                            print('sources (Turn 1)', step_id, sources)
                            prompt_instruction = copy.deepcopy(sources[0]['value'])
                            parts = split_and_clean(prompt_instruction)
                            content = []
                            for i in range(len(parts)):
                                if parts[i] == '<image>':
                                    content.append({'type': 'image', 'image': input_images[input_img_id]})
                                    input_img_id += 1
                                else:
                                    content.append({'type': 'text', 'text': parts[i]})

                            messages.append({'role': 'user', 'content': content})

                            past_key_values = DynamicCache(config=self.model.config)

                            # --- Turn 1 GPT: Generate tracking result ---
                            inputs_t1 = self.processor.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors='pt',
                            ).to(self.model.device)
                            step_input_tokens = int(inputs_t1['attention_mask'][0].sum().item())
                            episode_input_tokens_total += step_input_tokens
                            # episode_input_tokens_per_step.append(step_input_tokens)

                            with torch.no_grad():
                                output_ids_t1 = self.model.generate(
                                    **inputs_t1,
                                    max_new_tokens=4096,
                                    do_sample=False,
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                )

                            tracking_output = self.processor.tokenizer.decode(
                                output_ids_t1[0][inputs_t1.input_ids.shape[1]:],
                                skip_special_tokens=True
                            )
                            print('step_id:', step_id, 'tracking output:', tracking_output)

                            messages.append({
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': tracking_output}]
                            })

                            # --- Turn 2 Human: Decision instruction with current frame image ---
                            decision_parts = split_and_clean(self.decision_instruction)
                            decision_content = []
                            for part in decision_parts:
                                if part == '<image>':
                                    decision_content.append({'type': 'image', 'image': current_image})
                                else:
                                    decision_content.append({'type': 'text', 'text': part})
                            messages.append({
                                'role': 'user',
                                'content': decision_content
                            })

                            delta_messages = [
                                {'role': 'user', 'content': decision_content}
                            ]
                            delta_inputs = self.processor.apply_chat_template(
                                delta_messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors='pt',
                            ).to(self.model.device)
                            step_input_tokens = int(delta_inputs['attention_mask'][0].sum().item())
                            episode_input_tokens_total += step_input_tokens
                            # episode_input_tokens_per_step.append(step_input_tokens)

                            t1_cache_len = past_key_values.get_seq_length()
                            delta_seq_len = delta_inputs['input_ids'].shape[1]

                            # --- Turn 2 GPT: Generate decision (action) ---
                            max_retry = 10
                            llm_outputs = None
                            print(messages)
                            for retry_i in range(max_retry):
                                # if retry_i > 0:
                                #     past_key_values.crop(t1_cache_len)

                                # past_len = past_key_values.get_seq_length()
                                # delta_inputs['attention_mask'] = torch.cat([
                                #     torch.ones(1, past_len, dtype=delta_inputs['attention_mask'].dtype,
                                #                device=delta_inputs['attention_mask'].device),
                                #     delta_inputs['attention_mask'][:, -delta_seq_len:]
                                # ], dim=1)

                                # cache_position = torch.arange(
                                #     past_len, past_len + delta_inputs['input_ids'].shape[1],
                                #     device=self.model.device
                                # )
                                # with torch.no_grad():
                                    # output_ids_t2 = self.model.generate(
                                    #     **delta_inputs,
                                    #     max_new_tokens=2048,
                                    #     do_sample=not agent_found_target,
                                    #     temperature=0.7 if not agent_found_target else None,
                                    #     top_p=0.9 if not agent_found_target else None,
                                    #     use_cache=True,
                                    #     past_key_values=past_key_values,
                                    #     cache_position=cache_position,
                                    # )
                                # llm_outputs = self.processor.tokenizer.decode(
                                #     output_ids_t2[0][delta_inputs['input_ids'].shape[1]:],
                                #     skip_special_tokens=True
                                # )
                                # if step_id > 100:
                                #     os.makedirs("./_trash/tmp", exist_ok=True)
                                #     save_first_message_images_once(messages, output_dir="./_trash/tmp")
                                #     shutil.rmtree("./_trash/tmp", ignore_errors=True)
                                verify_inputs = self.processor.apply_chat_template(
                                    messages,
                                    tokenize=True,
                                    add_generation_prompt=True,
                                    return_dict=True,
                                    return_tensors="pt",
                                ).to(self.model.device)
                                with torch.no_grad():
                                    verify_output_ids = self.model.generate(
                                        **verify_inputs,
                                        max_new_tokens=2048,
                                        do_sample=not agent_found_target,
                                        temperature=0.7 if not agent_found_target else None,
                                        top_p=0.9 if not agent_found_target else None,
                                    )
                                llm_outputs = self.processor.tokenizer.decode(
                                    verify_output_ids[0][verify_inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True,
                                )
                                # if llm_outputs_verify == llm_outputs:
                                #     print(f'[VERIFY] step_id={step_id}: ALIGNED')
                                # else:
                                #     print(f'[VERIFY] step_id={step_id}: MISALIGNED!')
                                #     print(f'  cache  output: {llm_outputs}')
                                #     print(f'  message output: {llm_outputs_verify}')
                                if ('<move>' in llm_outputs) or ('<talk>' in llm_outputs):
                                    break
                            else:
                                print(f'WARNING: step_id {step_id} exhausted {max_retry} retries without valid action')

                            messages.append({
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': llm_outputs}]
                            })
                            print('step_id:', step_id, 'output text:', llm_outputs)
                        # if '<talk>' not in llm_outputs:
                        #     pre_decision_step = step_id
                        self._append_action_log(
                            scene_id,
                            episode_id,
                            [llm_outputs],
                            step_id=step_id,
                        )

                        # if '<talk>' not in llm_outputs and extract_question_Guaranteed(llm_outputs) is not None:
                        #     llm_outputs = add_talk_before_question(llm_outputs)

                        bbox_blocks, q_list, label_set = merge_tracking_and_current_outputs(
                            tracking_output=tracking_output,
                            llm_outputs=llm_outputs,
                            instance_type=instance_type,
                            messages=messages,
                            num_tracking_frames=len(tracking_images),
                        )
                        if self.save_video and bbox_blocks:
                            cur_images = tracking_images + [current_image]
                            if len(bbox_blocks) == len(cur_images):
                                cur_to_rgb = [next(i for i, img in enumerate(raw_rgb_list) if img is ci) for ci in cur_images]
                                for bbox_idx, bboxes in enumerate(bbox_blocks):
                                    image_id = cur_to_rgb[bbox_idx]
                                    image_with_bbox = cur_images[bbox_idx].resize((self.W, self.H))
                                    if bboxes:
                                        image_with_bbox, _ = draw_bboxes(image_with_bbox, bboxes, is_norm=True)
                                    step_id2track[image_id] = image_with_bbox
                        if '<talk>' in llm_outputs:
                            matched: Dict[str, Tuple[int, Dict[str, Any]]] = {}  # label -> (frame_idx, item)
                            npc_answer = None
                            matched = match_labels_min_frames(bbox_blocks, label_set, pre_decision_step + tracking_offset, max_frames=8)
                            frame_ground = [[] for _ in range(len(bbox_blocks))]
                            
                            detected = False
                            target_name = None
                            if not agent_found_target:
                                for lb in matched.values():
                                    frame_ground[lb[0]-pre_decision_step-tracking_offset].append(lb[1])
                                if len(frame_ground) > len(cur_segs):
                                    frame_ground = frame_ground[-len(cur_segs):]
                                    
                                for frame_idx in range(len(frame_ground)):
                                    best = (-1.0, None)
                                    instances = frame_ground[frame_idx]
                                    if instances:
                                        try:
                                            seg_map = cur_segs[frame_idx]   
                                        except:
                                            print('[error:]:length of ground is', len(frame_ground), 'but length of segs is', len(cur_segs))
                                        gt_box = mask01_to_bbox(seg_map)
                                        if gt_box:
                                            denorm_bbox_list = denorm_bbox(instances, seg_map, scale=1000.0, clamp=True)
                                            scores = []
                                            for item_idx, item in enumerate(instances):
                                                iou = IoU(denorm_bbox_list[item_idx]["bbox_2d"], gt_box)
                                                lb = denorm_bbox_list[item_idx].get("label")
                                                if iou > best[0]:
                                                    best = (iou, lb)

                                            if best[0] > 0.05:
                                                detected = True
                                                target_name = best[1]
                                                confirmed_target_label = target_name
                                                cur_iou = best_iou_for_target_label(
                                                    bbox_list=bbox_blocks[frame_idx],
                                                    seg_map=cur_segs[frame_idx],
                                                    target_label=confirmed_target_label,
                                                )
                                                if cur_iou is not None:
                                                    tracking_iou_history.append(float(cur_iou))
                                                    tracking_step_history.append(int(frame_idx + pre_decision_step + tracking_offset))

                                        if detected:
                                            for _idx, frame in enumerate(bbox_blocks):
                                                if any(obj.get("label") == target_name for obj in frame) and q_list[_idx] is not None:
                                                    frame_idx = _idx
                                            
                                            agent_found_target = True
                                            exp2nav_step = frame_idx + pre_decision_step + tracking_offset
                                            current_instances = [obj for obj in bbox_blocks[frame_idx] if obj.get("label") == target_name]
                                            bbox = bbox_blocks[frame_idx]
                                            bbox = sorted(bbox, key=lambda d: d["bbox_2d"][0])
                                            # gt_idx = next((i for i, d in enumerate(bbox) if d.get("label") == best[1]), -1)
                                            num_bbox = len(bbox)
                                            question = q_list[frame_idx]
                                            unknwon_count = sum(1 for x in bbox if x['label'] in label_set)
                                            # assert question is not None, "bug:question is none!"
                                            if num_bbox > 1:
                                                npc_answer = f"{target_name}."
                                            else:
                                                # question = f"Is this the {instance_type} you're looking for?"
                                                npc_answer = "Yes."
                                            bbox_message = format_with_tabs(bbox)
                                            think = (
                                                f"<think>I am looking for a specific {instance_type}. "
                                                f"In this image, I detected {num_bbox} {instance_type}(s).\n"
                                                f"{bbox_message}"
                                                f" {num_bbox-unknwon_count} of them have been checked and are not the target instance."
                                                f" The remaining {unknwon_count} are still uncertain.</think>"
                                            )
                                            # dialogs = []
                                            dialogs = remove_instance_qa(dialogs, target_name)
                                            dialogs.append({'role': 'navigator', 'message': question, 'true_idx': frame_idx + pre_decision_step + 1 + tracking_offset, 'thinking': think})
                                            dialogs.append({'role': 'oracle', 'message': npc_answer, 'true_idx': frame_idx + pre_decision_step + 1 + tracking_offset})
                                            
                                            self._append_action_log(
                                                scene_id,
                                                episode_id,
                                                [question, npc_answer],
                                                step_id=step_id,
                                            )
                                            frame_tag = "Frame_c" if (frame_idx == len(q_list) - 1 and q_list[-1] is not None) else f"Frame_{frame_idx + 1}"
                                            npc_answers = f"{frame_tag}: {npc_answer} "
                                            break

                                if not detected:
                                    npc_answers = ''
                                    for frame_idx, q_ in enumerate(q_list):
                                        if q_:
                                            q_res = match_utterance(q_, TEMPLATES_TEST)
                                            assert q_res is not None, f"{q_} !!!"

                                            if q_res['scene_type'] == 'single_candidate':
                                                npc_answer = "No."
                                            elif q_res['scene_type'] == 'multi_all_unseen':
                                                npc_answer = "None of them."
                                            elif q_res['scene_type'] == "multi_confirmed_partial_except":
                                                npc_answer = "No."

                                            bbox = bbox_blocks[frame_idx]
                                            bbox = sorted(bbox, key=lambda d: d["bbox_2d"][0])
                                            num_bbox = len(bbox)
                                            bbox_message = format_with_tabs(bbox)

                                            if not q_res and num_bbox == 1:
                                                npc_answer = "No."
                                            elif not q_res and num_bbox > 1:
                                                npc_answer = "None of them."

                                            appeared_labels = {x['label'] for x in bbox if x['label'] in label_set}
                                            unknown_count = len(appeared_labels)
                                            label_set.difference_update(appeared_labels)
                                            think = (
                                                f"<think>I am looking for a specific {instance_type}. "
                                                f"In this image, I detected {num_bbox} {instance_type}(s).\n"
                                                f"{bbox_message}"
                                                f" {num_bbox-unknown_count} of them have been checked and are not the target instance."
                                                f" The remaining {unknown_count} are still uncertain.</think>"
                                            )
                                            dialogs.append({
                                                'role': 'navigator',
                                                'message': q_,
                                                'true_idx': frame_idx + pre_decision_step + 1 + tracking_offset,
                                                'thinking': think
                                            })
                                            dialogs.append({
                                                'role': 'oracle',
                                                'message': npc_answer,
                                                'true_idx': frame_idx + pre_decision_step + 1 + tracking_offset
                                            })

                                            self._append_action_log(
                                                scene_id,
                                                episode_id,
                                                [q_, npc_answer],
                                                step_id=step_id,
                                            )

                                            frame_tag = "Frame_c" if (frame_idx == len(q_list) - 1 and q_list[-1] is not None) else f"Frame_{frame_idx + 1}"
                                            npc_answers += f"{frame_tag}: {npc_answer}"
                            action = 6
                            continue

                        else:
                            bbox_blocks_ = extract_bbox_blocks_from_chat_sources(llm_outputs)
                            if bbox_blocks_:
                                bbox_blocks = bbox_blocks_
                            if bbox_blocks:
                                if not agent_found_target:
                                    pass
                                    # fallback_best = (-1.0, None, None)  # (iou, label, frame_idx)
                                    # for frame_idx, bbox in enumerate(bbox_blocks):
                                    #     if not bbox:
                                    #         continue
                                    #     if frame_idx >= len(cur_segs):
                                    #         continue

                                    #     seg_map = cur_segs[frame_idx]
                                    #     gt_box = mask01_to_bbox(seg_map)
                                    #     if gt_box is None:
                                    #         continue

                                    #     denorm_bbox_list = denorm_bbox(
                                    #         bbox, seg_map, scale=1000.0, clamp=True
                                    #     )

                                    #     for item_idx, item in enumerate(bbox):
                                    #         pred_box = denorm_bbox_list[item_idx]["bbox_2d"]
                                    #         lb = denorm_bbox_list[item_idx].get("label")
                                    #         iou = IoU(pred_box, gt_box)

                                    #         if iou > fallback_best[0]:
                                    #             fallback_best = (iou, lb, frame_idx)

                                    # if fallback_best[0] > 0.2:
                                    #     target_name = fallback_best[1]
                                    #     frame_idx = fallback_best[2]
                                    #     confirmed_target_label = target_name

                                    #     agent_found_target = True
                                    #     exp2nav_step = frame_idx + pre_decision_step + tracking_offset

                                    #     bbox = bbox_blocks[frame_idx]

                                    #     dialogs = remove_instance_qa(dialogs, target_name)

                                    #     nav_dialog, oracle_dialog = build_updated_tracking_dialog(
                                    #         instance_type=instance_type,
                                    #         bbox=bbox,
                                    #         target_name=target_name,
                                    #         abs_frame_idx=frame_idx + pre_decision_step + 1 + tracking_offset,
                                    #     )

                                    #     dialogs.append(nav_dialog)
                                    #     dialogs.append(oracle_dialog)

                                    #     self._append_action_log(
                                    #         scene_id,
                                    #         episode_id,
                                    #         [nav_dialog["message"], oracle_dialog["message"]],
                                    #         step_id=step_id,
                                    #     )

                                else:
                                    for frame_idx, bbox in enumerate(bbox_blocks):
                                        if not bbox:
                                            continue
                                        if frame_idx >= len(cur_segs):
                                            continue

                                        cur_iou = best_iou_for_target_label(
                                            bbox_list=bbox,
                                            seg_map=cur_segs[frame_idx],
                                            target_label=confirmed_target_label,
                                        )

                                        if cur_iou is not None:
                                            tracking_iou_history.append(float(cur_iou))
                                            tracking_step_history.append(
                                                int(frame_idx + pre_decision_step + tracking_offset)
                                            )

                                bbox_blocks = None
                            # ========= fallback end =========
                            if '↓' not in llm_outputs:
                                pre_decision_step = step_id
                                output_tokens = int(verify_output_ids.shape[1])
                                episode_input_tokens_total += output_tokens
                                episode_input_tokens_per_step.append(output_tokens)

                            move_idx = llm_outputs.find("<move>")
                            llm_outputs = llm_outputs[move_idx:] if move_idx != -1 else llm_outputs
                            if bool(re.search(r'\d', llm_outputs)): # output pixel goal 
                                forward_action = 0
                                coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                                if self.args.use_prompt_image:
                                    pixel_goal = [coord[0] * image_size[1] // 9, coord[1] * image_size[0] // 9] 
                                else:
                                    try:
                                        print('coords:', coord)
                                        # check=True
                                        coord = self.denorm_pixel(coord)
                                        pixel_goal = [int(coord[1]), int(coord[0])]  # switch the goal o
                                    except:
                                        break
                                # trans pixel goal to global goal
                                try:
                                    goal = self.pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)
                                except:
                                    break
                                print('before', goal, depth.shape)
                                
                                draw = ImageDraw.Draw(save_raw_image, 'RGB')
                                x, y, r = pixel_goal[1], pixel_goal[0], 2
                                draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255,0,0))
                                # look down --> horizontal
                                env.step(4)
                                env.step(4)
                            
                                if look_down_image is not None:
                                    prev_look_image = look_down_image.resize((self.args.resize_w, self.args.resize_h))
                                local_type = self.args.local_planner
                                if local_type == 'shortest_path':
                                    goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3] 
                                    if not env.sim.pathfinder.is_navigable(np.array(goal)):
                                        goal = np.array(env.sim.pathfinder.snap_point(np.array(goal)))
                                    action = agent.get_next_action(goal)
                                elif local_type == 'pointnav':
                                    action = self._pointnav(np.array(goal), torch.from_numpy(observations['depth']).unsqueeze(0).cuda(), step_id, robot_xy, camera_yaw, stop=False)
                                elif local_type == 'DiT':
                                    local_actions = []
                                    pixel_values = inputs.pixel_values
                                    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                                    with torch.no_grad():
                                        traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                                    image_dp = (
                                        torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                    )
                                    pix_goal_image = copy.copy(image_dp)
                                    images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                    # depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                                    # pix_goal_depth = copy.copy(depth_dp)
                                    # depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                                    with torch.no_grad():
                                        dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp=None)
                                    action_list = traj_to_actions(dp_actions)
                                    if len(action_list) < MAX_STEPS:
                                        action_list += [0] * (MAX_STEPS - len(action_list))

                                    local_actions = action_list
                                    if len(local_actions) >= MAX_LOCAL_STEPS:
                                        local_actions = local_actions[:MAX_LOCAL_STEPS]

                                    action = local_actions[0]

                                if action == 0:
                                    goal = None
                                    output_ids = None
                                    action = 2
                                    print('conduct a random action 2')
                                    observations, _, _ = env.apply_action(action)
                                    # observations = {
                                    #         'rgb': observations.rgb,
                                    #         'depth': observations.depth,
                                    #         'compass': observations.compass,
                                    #         'gps': observations.gps,
                                    #         'semantic': observations.semantic
                                    #     }
                                    step_id += 1
                                    messages = []
                                    continue
                            
                                print('predicted goal', pixel_goal, goal, flush=True)
                            else:                           
                                action_seq = self.parse_actions(llm_outputs)
                                print('actions', action_seq, flush=True)
                

                    if len(action_seq) != 0:
                        action = action_seq[0]
                        action_seq.pop(0)
                    elif goal is not None:
                        if local_type == 'DiT':
                            if len(local_actions) == 0:
                                local_actions = []
                                image_dp = (
                                    torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                )

                                images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                # depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                                # depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp=None)

                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < MAX_STEPS:
                                    action_list += [0] * (MAX_STEPS - len(action_list))

                                local_actions = action_list
                                if len(local_actions) >= MAX_LOCAL_STEPS:
                                    local_actions = local_actions[:MAX_LOCAL_STEPS]
                                print("local_actions", local_actions)
                                action = local_actions.pop(0)
                            else:
                                action = local_actions.pop(0)
                        else:
                            if local_type == 'shortest_path':
                                action = agent.get_next_action(goal)
                            elif local_type == 'pointnav':
                                action = self._pointnav(np.array(goal), torch.from_numpy(observations['depth']).unsqueeze(0).cuda(), step_id, robot_xy, camera_yaw, stop=False)
                            action = self._to_scalar_action(action)
            
                        forward_action +=1
                        print('forward_action', forward_action, flush=True)
                        if forward_action > 8:
                            goal =  None
                            output_ids = None
                            messages = []
                            step_id += 1
                            forward_action =0
                            continue
                        if action == 0:
                            goal = None
                            output_ids = None
                            messages = []
                            step_id += 1
                            continue
                    else:
                        action = 0
                        
                    if info['top_down_map'] is not None:
                        if save_dot:
                            save_raw_image = self.dot_matrix_two_dimensional(save_raw_image, save_img=False, save_path=f'test_{step_id}.jpg', pixel_goal = pixel_goal)
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        frame = append_text_underneath_image(frame, f"step: {step_id}")
                        frame = append_text_underneath_image(frame, f"action: {llm_outputs}")
                        vis_frames.append(frame)
                        # if check:
                        #     check = False
                        plt.imsave(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'frame_{idx}.jpg'), frame)
                        
                    print("step_id", step_id, "action", action)
                    path_list.append(env.sim.get_agent_state().position.tolist())
                    action_list.append(int(action))
                    if action == 5:
                        env.step(action)
                        observations, _, _ = env.apply_action(action)
                    elif action == 0:
                        # observations, _, _ = env.apply_action(1)
                        # observations, _, _ = env.apply_action(1)
                        observations, _, _ = env.apply_action(1)
                        observations, _, _ = env.apply_action(1)
                        observations, _, _ = env.apply_action(action)
                        step_id += 1
                    else:
                        observations, _, _ = env.apply_action(action)
                        step_id += 1
                        messages = []
                
                process_bar.update(1)
                metrics = env.get_metrics()
                if self.save_video:
                    for idx_, frame in enumerate(vis_frames):
                        if idx_ in step_id2track:
                            track_img = step_id2track[idx_]  
                            track_img = track_img.resize((640, 480))
                            track_img_np = np.array(track_img)
                            track_img_np = cv2.cvtColor(track_img_np, cv2.COLOR_RGB2BGR)
                            vis_frames[idx_][:480, :640] = track_img_np
                    vis_frames = unify_to_first(vis_frames)
                    out_dir = os.path.join(self.output_path, f"vis_0", scene_id, f"{episode_id:04d}")
                    os.makedirs(out_dir, exist_ok=True)

                    for i, f in enumerate(vis_frames):
                        if not isinstance(f, Image.Image):
                            f = Image.fromarray(np.asarray(f))
                        f.save(os.path.join(out_dir, f"{i:06d}.png"))

                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                nes.append(metrics["distance_to_goal"])
                print(f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")

                failure_reason, reached_neighborhood, last_two_tracking_ious = classify_episode_failure(
                    success=bool(metrics["success"]),
                    ever_observed=bool(found_target),
                    ever_grounded=bool(agent_found_target),
                    tracking_iou_history=tracking_iou_history,
                    distance_to_goal=float(metrics["distance_to_goal"]),
                    track_iou_thr=0.2,
                    neighborhood_thr=1.0,
                )
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "found": (found_target,found_step),
                    "failure_reason": failure_reason,
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                    "path": path_list,
                    "action": action_list,
                    "object_category": episode.object_category if 'vln' not in self.args.task else '',
                    "episode_input_tokens_total": sum(episode_input_tokens_per_step)/len(episode_input_tokens_per_step),
                }
                self._append_result(result)
                # exit()
        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(nes).to(self.device), torch.tensor(len(sucs)).to(self.device)

    def parse_actions(self, output):
        if not output:
            return []
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        matches = re.findall(action_patterns, output)
        return [self.actions2idx[m][0] for m in matches]

    def preprocess_qwenvl(self, source):
        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
        if source[0]["value"]:
            source[0]["value"] += f" {prompt}."
        else: 
            source[0]["value"] = f"{prompt}."
        return source


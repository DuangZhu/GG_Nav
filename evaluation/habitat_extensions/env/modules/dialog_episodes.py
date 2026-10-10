#!/usr/bin/env python3
import numpy as np
from typing import List, Optional, Union

import attr
from habitat.core.dataset import Episode
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal

@attr.s(auto_attribs=True)
class AgentPosition:
    position: Union[None, List[float], np.ndarray]

@attr.s(auto_attribs=True)
class DialogViewLocation:
    agent_state: AgentPosition

@attr.s(auto_attribs=True, kw_only=True)
class DialogImageGoal:
    position: List[float] = attr.ib(default=None, validator=not_none_validator)

    # SPL 里 image_goal 会带 rotation（四元数）
    rotation: Optional[List[float]] = None

    frame_coverage: Optional[float] = None
    object_coverage: Optional[float] = None
    hfov: Optional[float] = None
    image_dimensions: Optional[List[int]] = None

@attr.s(auto_attribs=True, kw_only=True)
class DialogGoal(NavigationGoal):
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None
    bbox: Optional[List[float]] = None
    view_points: Optional[List[DialogViewLocation]] = None

    # for spl data
    object_id: Optional[int] = None
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    object_surface_area: Optional[float] = None
    image_goals: Optional[List[DialogImageGoal]] = None
    
    
@attr.s(auto_attribs=True, kw_only=True)
class DialogEpisode(Episode):
    object_category: Optional[str] = None
    goals: List[DialogGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    instruction: Optional[dict] = []
    frames: Optional[list] = []
    # for spl data
    goal_object_id: Optional[str] = None
    goal_image_id: Optional[int] = None 
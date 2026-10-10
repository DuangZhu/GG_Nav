from typing import Optional
import habitat
import gym
from habitat.core.dataset import Dataset, Episode
import numpy as np
from habitat.core.simulator import Observations
from habitat.core.env import Env
from omegaconf import DictConfig
from habitat.config.default import get_agent_config
from depth_camera_filtering import filter_depth
from .mp3d_gt import MP3DGTPerception
from .interfaces import SemObservations
from .constants import (
    HM3D_6Cat,
    OVON_Cat,
)
import quaternion
from typing import Any, Dict, Optional, Tuple

@habitat.registry.register_env(name="GymObjectNavEnv")
class GymObjectNavEnv(gym.Wrapper):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None, exp_config=None
    ):
        base_env = ObjectNavEnv(config=config, dataset=dataset, exp_config=exp_config)
        super().__init__(base_env)

class ObjectNavEnv(Env):
    # ObjectNav-family tasks use object_category episodes (no .instruction).
    # VLN / dialog tasks go through _preprocess_obs which reads episode.instruction.
    _OBJECT_TASKS = frozenset({
        'object', 'objectnav', 'psl', 'ovon',
        'object_pi', 'objectnav_pi',
        'object_xr1', 'objectnav_xr1',
        'object_g05', 'objectnav_g05',
        'object_pi_flow'
    })

    def __init__(
        self,
        config: "DictConfig",
        dataset: Optional[Dataset[Episode]] = None,
        exp_config=None,
    ):
        super().__init__(config, dataset)
        agent_config = get_agent_config(config.simulator)
        self.min_depth = agent_config.sim_sensors.depth_sensor.min_depth
        self.max_depth = agent_config.sim_sensors.depth_sensor.max_depth
        self._camera_fov = np.deg2rad(agent_config.sim_sensors.depth_sensor.hfov)
        self._fx = self._fy = agent_config.sim_sensors.depth_sensor.width / (2 * np.tan(self._camera_fov / 2))
        self._camera_height = agent_config.sim_sensors.rgb_sensor.position[1]
        self.episodes_data_path = config.dataset.data_path
            
        self.one_hot_encoding = np.eye(
            exp_config.num_sem_categories
        )
        if exp_config.vocabulary == 'hm3d':  
            self.semantic_category_mapping = HM3D_6Cat()
        elif exp_config.vocabulary == 'ovon':
            self.semantic_category_mapping = OVON_Cat()
        else:
            raise NotImplementedError
        if exp_config.detector == 'mp3d_gt':
            self.segmentation = MP3DGTPerception(self.max_depth, self.min_depth, self._fx, self._fy)
        else:
            raise NotImplementedError

        self.config = config
        self.exp_config = exp_config

    def reset(self):
        obs = super().reset()
        if self.exp_config.task in self._OBJECT_TASKS:
            objects = self.sim.semantic_annotations().objects
            self.semantic_category_mapping.reset_instance_id_to_category_id(objects)
            self._last_obs = self._preprocess_obs_obj(obs)
        else:
            self._last_obs = self._preprocess_obs(obs)
        observations = {
                        'rgb': self._last_obs.rgb,
                        'depth': self._last_obs.depth,
                        'compass': self._last_obs.compass,
                        'gps': self._last_obs.gps,
                        'semantic': self._last_obs.semantic
                        }
        return observations

    def get_tf_episodic_to_global(self):
        agent_state = self.sim.get_agent_state()
        rotation = agent_state.rotation
        translation = agent_state.position
        rotation_matrix = quaternion.as_rotation_matrix(rotation)
        tf_episodic_to_global = np.eye(4)
        tf_episodic_to_global[:3, :3] = rotation_matrix
        tf_episodic_to_global[:3, 3] = translation
        return tf_episodic_to_global

    def apply_action(
        self,
        action,
    ):
        if self.episode_over:
            metric = self.get_metrics()
            observations = {
                "rgb": self._last_obs.rgb,
                "depth": self._last_obs.depth,
                "compass": self._last_obs.compass,
                "gps": self._last_obs.gps,
                "semantic": self._last_obs.semantic,
            }
            return observations, True, metric
            
        obs = self.step(action)
        self._last_obs = (
            self._preprocess_obs_obj(obs)
            if self.exp_config.task in self._OBJECT_TASKS
            else self._preprocess_obs(obs)
        )
        metric = self.get_metrics()
        observations = {
                'rgb': self._last_obs.rgb,
                'depth': self._last_obs.depth,
                'compass': self._last_obs.compass,
                'gps': self._last_obs.gps,
                'semantic': self._last_obs.semantic
                }
        return observations, self.episode_over, metric
    
    def _preprocess_obs(self, obs: Observations):
        goal_name = self.current_episode.instruction.instance_id[0].split('/')[-1].split('_')[0]
        task_observations={
                "goal_name": goal_name
            }
        sem_obs = SemObservations(
                rgb= obs["rgb"], 
                depth=obs["depth"],
                compass=obs["compass"],
                gps=obs["gps"],
                task_observations=task_observations,
                camera_pose=None,
                third_person_image=None
            )
        targets = [self.current_episode.goals[idx].bbox for idx, _ in enumerate(self.current_episode.instruction.instance_id)]
        targets = np.array([[target[0], min(-target[2], -target[5]) , target[1], target[3], max(-target[5], -target[2]), target[4]] for target in targets])
        depth = filter_depth(sem_obs.depth.reshape(sem_obs.depth.shape[:2]), blur_type=None)
        tf_camera_to_global = self.get_tf_episodic_to_global()
        tf_camera_to_global[1, 3] = self._camera_height + self.sim.get_agent_state().position[1]
        tf_camera_to_ply = np.dot(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), tf_camera_to_global)
        sem_obs.semantic = self.segmentation.predict(depth, targets, tf_camera_to_ply)
        return sem_obs
    
    def _preprocess_obs_obj(self, obs: Observations):
        task_observations={}

        if self.exp_config.goal_map_only:
            task_observations["object_goal"] = 0
        
        sem_obs = SemObservations(
                rgb= obs["rgb"], 
                depth=obs["depth"],
                compass=obs["compass"],
                gps=obs["gps"],
                task_observations=task_observations,
                camera_pose=None,
                third_person_image=None
            )
        if 'semantic' not in obs:
            obs["semantic"] = np.zeros((obs["rgb"].shape[0], obs["rgb"].shape[1], 1))
        sem_obs = self._preprocess_semantic(sem_obs, obs["semantic"])
        return sem_obs
    
    def _preprocess_semantic(self, obs:Observations, habitat_semantic: np.ndarray):
        goal_ids = [i.object_id for i in self.current_episode.goals]
        semantic_images = []
        for i in goal_ids:
            semantic_image = habitat_semantic[:,:,0] == i
            semantic_images.append(semantic_image)
        if len(semantic_images) > 0:
            semantic_images = np.stack(semantic_images, axis=0)
        else:
            semantic_images = np.zeros((1, habitat_semantic.shape[0], habitat_semantic.shape[1]), dtype=np.uint8)
        obs.semantic = semantic_images
        return obs

    def _preprocess_goal(self, goal: np.array) -> Tuple[int, str]:
        return self.semantic_category_mapping.map_goal_id(goal[0])
    
    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info and self.visualize:
            info["semantic_category_mapping"] = self.semantic_category_mapping
    
    @property
    def original_action_space(self):
        return self.action_space

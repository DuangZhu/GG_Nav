
import gzip
import json
import os
import attr
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import copy

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from .dialog_episodes import DialogEpisode, DialogGoal, DialogViewLocation, AgentPosition
if TYPE_CHECKING:
    from omegaconf import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

@attr.s(auto_attribs=True, kw_only=True)
class DialogInstructionData:
    task_type: str
    instruction_text: str
    instance_id: List[str]
    instruction_info: Optional[List[str]] = None

@registry.register_dataset(name="dialog")
class DialogDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[DialogEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)
        # import ipdb;ipdb.set_trace()
        if not self.episodes:
            dataset_path = Path(dataset_filename).resolve()
            content_dir = dataset_path.parent / "content"

            if content_dir.is_dir():
                for p in sorted(content_dir.glob("*.json.gz")):
                    with gzip.open(p, "rt", encoding="utf-8") as f:
                        self._from_json_append(f.read(), scenes_dir=config.scenes_dir)
        # import ipdb;ipdb.set_trace()
        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def _from_json_append(self, json_str: str, scenes_dir: str) -> None:
        """
        兼容两种 from_json 实现：
        1) from_json 直接 append 到 self.episodes（推荐实现）
        2) from_json 内部会 self.episodes = [...] 覆盖（也能被这里修正为“追加”）
        """
        old_list = self.episodes
        old_id = id(old_list)
        old_copy = list(old_list)

        self.from_json_spl(json_str, scenes_dir=scenes_dir)
        
        # 如果 from_json 把 self.episodes 换成了新 list，则把旧的接回去，实现“追加”
        if id(self.episodes) != old_id:
            self.episodes = old_copy + list(self.episodes)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        # import ipdb;ipdb.set_trace()
        # self.instruction_vocab = VocabDict(
        #     word_list=deserialized["instruction_vocab"]["word_list"]
        # )
        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]

        for episode in deserialized["episodes"]:
            episode = DialogEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = DialogInstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                view_points = []
                for view_point in goal['view_points']:
                    view_point = DialogViewLocation(**{'agent_state': AgentPosition(**view_point['agent_state'])})
                    view_points.append(view_point)
                goal['view_points'] = view_points
                episode.goals[g_index] = DialogGoal(**goal)
            self.episodes.append(episode)

    def from_json_spl(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        # import ipdb;ipdb.set_trace()
        # self.instruction_vocab = VocabDict(
        #     word_list=deserialized["instruction_vocab"]["word_list"]
        # )
        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]

        for episode in deserialized["episodes"]:
            episode_ = copy.deepcopy(episode)
            episode = DialogEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = DialogInstructionData(task_type='instance', instruction_text='Search for the '+episode_['object_category']+'.', instance_id=[episode_['goal_object_id']])
            
            episode_name = episode_['scene_id'].split('/')[-1].split('.')[0] + '_' + str(episode_['goal_object_id'])
            episode.goals = [copy.deepcopy(deserialized["goals"][episode_name])]
            # print(episode_name)
            
            for g_index, goal in enumerate(episode.goals):
                view_points = []
                
                for view_point in goal['view_points']:
                    view_point["agent_state"].pop("rotation", None)
                    view_point = DialogViewLocation(**{'agent_state': AgentPosition(**view_point['agent_state'])})
                    view_points.append(view_point)

                goal['view_points'] = view_points
                episode.goals[g_index] = DialogGoal(**goal)
            self.episodes.append(episode)
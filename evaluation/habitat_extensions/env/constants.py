# ------------------------------------------------------------------------------
# Original code from https://github.com/facebookresearch/home-robot
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified by: Meng Wei
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import json
import itertools
import matplotlib

hm3d_to_mp3d_path = Path(__file__).resolve().parent / "matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep="    ", header=0, engine="python")
hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}
hm3d_category_name_to_mp3d_category_name = {row["category"]: row["mpcat40"] for _, row in df.iterrows()}

# Color constants we use.
# Note: originally from Habitat
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

class SemanticCategoryMapping(ABC):
    """
    This class contains a mapping from semantic and goal category IDs provided by
    a Habitat environment to category IDs stored in the semantic map, as well as
    the color palettes and legends to visualize these categories.
    """

    def __init__(self):
        pass

    @abstractmethod
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        pass

    @abstractmethod
    def reset_instance_id_to_category_id(self, env):
        """Reset instance id. Env should be a simulation environment."""
        pass

    @property
    @abstractmethod
    def instance_id_to_category_id(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def map_color_palette(self):
        pass

    @property
    @abstractmethod
    def frame_color_palette(self):
        pass

    @property
    @abstractmethod
    def categories_legend_path(self):
        pass

    @property
    @abstractmethod
    def num_sem_categories(self):
        pass

    @property
    def num_sem_obj_categories(self):
        return self.num_sem_categories()
    
# COCO Indoor Categories

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "no-category": 15,
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

coco_categories_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]

coco_categories_legend_path = str(
    Path(__file__).resolve().parent / "coco_categories_legend.png"
)

coco_frame_color_palette = [
    int(x * 255.0)
    for x in [
        *coco_categories_color_palette,
        1.0,
        1.0,
        1.0,  # no category
    ]
]

coco_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.0,
        0.0,
        0.0,  # instance border
        *coco_categories_color_palette,
    ]
]
mp3d_to_coco = {
    3: 0,  # chair
    10: 1,  # couch
    14: 2,  # plant
    11: 3,  # bed
    18: 4,  # toilet
    22: 5,  # tv
    5: 6,  # table
    15: 8,  # sink
}

class HM3DtoCOCOIndoor(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to COCO indoor
    category IDs.
    """

    def __init__(self):
        super().__init__()
        self.goal_id_to_goal_name = {idx: name for name, idx in coco_categories.items()}
        self.hm3d_goal_id_to_coco_goal_name = {
            0: "chair",
            1: "bed",
            2: "plant",
            3: "toilet",
            4: "tv monitor",
            5: "sofa",
        }
        self.hm3d_goal_id_to_coco_goal_id = {
            0: 0,  # chair
            1: 3,  # bed
            2: 2,  # potted plant
            3: 4,  # toilet
            4: 5,  # tv
            5: 1,  # couch
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_coco_goal_id[goal_id],
            self.hm3d_goal_id_to_coco_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                mp3d_to_coco.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip()),
                    self.num_sem_categories - 1,
                )
                for obj in objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return coco_map_color_palette

    @property
    def frame_color_palette(self):
        return coco_frame_color_palette

    @property
    def categories_legend_path(self):
        return coco_categories_legend_path

    @property
    def num_sem_categories(self):
        return 16

# hm3d 9 Indoor Categories

hm3d_9cat_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # sofa
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv monitor
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # fireplace
    0.66,
    0.9232,
    0.9400000000000001,  # bathtub
    0.66,
    0.8182,
    0.9400000000000001,  # mirror
]

hm3d_9cat_frame_color_palette = [
    int(x * 255.0)
    for x in [
        *hm3d_9cat_color_palette,
        1.0,
        1.0,
        1.0,  # no category
    ]
]

hm3d_9cat_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.0,
        0.0,
        0.0,  # instance border
        *hm3d_9cat_color_palette,
    ]
]

# MP3D Categories
mp3d_21cats = {
            'chair':0, #g
            'table':1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'unknown':21
        }

mp3d_category_color_palette = [
    0.9400,
    0.7818,
    0.66,
    0.12,
    0.7495,
    0.2888,
    0.8909,
    0.222,
    0.5654,
    0.6724,
    0.9213,
    0.0769,
    0.4402,
    0.5997,
    0.9882,
    0.8317,
    0.9637,
    0.2399,
    0.1875,
    0.7772,
    0.7753,
    0.9499,
    0.4683,
    0.8045,
    0.5668,
    0.1212,
    0.9877,
    0.8493,
    0.7415,
    0.1735,
    0.2998,
    0.8316,
    0.9572,
    0.9965,
    0.3557,
    0.6005,
    0.6441,
    0.9754,
    0.2826,
    0.4273,
    0.6026,
    0.9916,
    0.7846,
    0.9588,
    0.2101,
    0.1613,
    0.8475,
    0.7562,
    0.9298,
    0.2867,
    0.7787,
    0.5224,
    0.0847,
    0.9835,
    0.8851,
    0.6833,
    0.1429,
    0.2467,
    0.8725,
    0.9484,
    0.9952,
    0.5337,
    0.678
]

mp3d_frame_color_palette = [
    int(x * 255.0)
    for x in [
        *mp3d_category_color_palette,
        1.0,
        1.0,
        1.0,  # no category
    ]
]
mp3d_map_color_palette = [
     int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.0,
        0.0,
        0.0,  # instance border
        *mp3d_category_color_palette,
    ]
]

# hm3d 92 Indoor Categories
def getCMap(labelCount, cmapName='jet'):
    # Get jet color map from Matlab
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    return cmap

hm3d_92cat_color_palette = list(itertools.chain.from_iterable(getCMap(92)))

hm3d_92cat_frame_color_palette = [
    int(x * 255.0)
    for x in [
        *hm3d_92cat_color_palette,
        1.0,
        1.0,
        1.0,  # no category
    ]
]

hm3d_92cat_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.0,
        0.0,
        0.0,  # instance border
        *hm3d_92cat_color_palette,
    ]
]

class MP3D_21Cat(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self._instance_id_to_category_id = None
        self.hab_goal_id_to_mp3d_goal_name = {
            0: "chair",
            1: "table",
            2: "picture",
            3: "cabinet",
            4: "cushion",
            5: "sofa",
            6:'bed',
            7:'chest_of_drawers',
            8:'plant',
            9:'sink',
            10:'toilet',
            11:'stool',
            12:'towel',
            13:'tv_monitor',
            14:'shower',
            15:'bathtub',
            16:'counter',
            17:'fireplace',
            18:'gym_equipment',
            19:'seating',
            20:'clothes'
        }
        self.hab_goal_id_to_mp3d_goal_id = {i:i for i in range(len(self.hab_goal_id_to_mp3d_goal_name))}

    @property
    def map_color_palette(self):
        return mp3d_map_color_palette

    @property
    def frame_color_palette(self):
        return mp3d_frame_color_palette

    @property
    def categories_legend_path(self):
        return None
        
    @property
    def num_sem_categories(self):
        return len(mp3d_21cats)
    
    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                mp3d_21cats.get(
                    obj.category.name().lower().strip(), 
                        self.num_sem_categories - 1,)
                for obj in objects
            ]
        )
    
    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id
    
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hab_goal_id_to_mp3d_goal_id[goal_id],
            self.hab_goal_id_to_mp3d_goal_name[goal_id],
        )

class HM3D_92Cat(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to 92 indoor
    category IDs.
    """
    def __init__(self):
        super().__init__()
        self.hm3d_92cats = json.load(open('data/hm3d_92cat.json','r'))
        self.hm3d_cat_name_to_cat_id = {x:idx for idx,x in enumerate(self.hm3d_92cats)}
        self.hm3d_mapping = json.load(open('data/hm3d92_mapping.json','r'))
        self._instance_id_to_category_id = None
        self.hm3d_goal_id_to_cat90_goal_name = {
            0: "chair",
            1: "bed",
            2: "potted plant",
            3: "toilet",
            4: "tv",
            5: "sofa",
        }
        self.hm3d_goal_id_to_cat90_goal_id = {
            0: 0,  # chair
            1: 3,  # bed
            2: 2,  # potted plant
            3: 4,  # toilet
            4: 5,  # tv monitor
            5: 1,  # sofa
        }

    @property
    def map_color_palette(self):
        return hm3d_92cat_map_color_palette

    @property
    def frame_color_palette(self):
        return hm3d_92cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return None
        
    @property
    def num_sem_categories(self):
        return len(self.hm3d_92cats)
    
    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                self.hm3d_cat_name_to_cat_id.get(
                    self.hm3d_mapping.get(obj.category.name().lower().strip(), 'unknown'),
                    self.num_sem_categories - 1,
                )
                for obj in objects
            ]
        )
    
    def get_goal_ids(self):
        return sorted(list(itertools.chain.from_iterable(self.hm3d_goal_id_to_cat90_goal_id.values())))
    
    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id
    
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_cat90_goal_id[goal_id],
            self.hm3d_goal_id_to_cat90_goal_name[goal_id],
        )

class HM3D_9Cat(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to 10 indoor
    category IDs.
    """

    def __init__(self):
        super().__init__()
        self.hm3d_9cats = ["chair", "sofa", "plant", "bed", "toilet", "tv monitor", "fireplace", "bathtub", "mirror", "unknown"]
        self.hm3d_cat_name_to_cat_id = {x:idx for idx,x in enumerate(self.hm3d_9cats)}
        self.hm3d_goal_id_to_goal_name = {
            0: "chair",
            1: "bed",
            2: "plant",
            3: "toilet",
            4: "tv monitor",
            5: "sofa",
        }
        self.hm3d_goal_id_to_goal_id = {
            0: 0,  # chair
            1: 3,  # bed
            2: 2,  # plant
            3: 4,  # toilet
            4: 5,  # tv
            5: 1,  # sofa
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_goal_id[goal_id],
            self.hm3d_goal_id_to_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                self.hm3d_cat_name_to_cat_id.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip(), 'unknown'),
                    self.num_sem_categories - 1,
                )
                for obj in objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return hm3d_9cat_map_color_palette

    @property
    def frame_color_palette(self):
        return hm3d_9cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return None

    @property
    def num_sem_categories(self):
        return len(self.hm3d_9cats)


class HM3D_6Cat(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to 10 indoor
    category IDs.
    """

    def __init__(self):
        super().__init__()
        self.hm3d_9cats = ["chair", "sofa", "plant", "bed", "toilet", "tv monitor"]
        self.hm3d_cat_name_to_cat_id = {x:idx for idx,x in enumerate(self.hm3d_9cats)}
        self.hm3d_goal_id_to_goal_name = {
            0: "chair",
            1: "bed",
            2: "plant",
            3: "toilet",
            4: "tv monitor",
            5: "sofa",
        }
        self.hm3d_goal_id_to_goal_id = {
            0: 0,  # chair
            1: 3,  # bed
            2: 2,  # plant
            3: 4,  # toilet
            4: 5,  # tv
            5: 1,  # sofa
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_goal_id[goal_id],
            self.hm3d_goal_id_to_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                self.hm3d_cat_name_to_cat_id.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip(), 'unknown'),
                    self.num_sem_categories - 1,
                )
                for obj in objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return hm3d_9cat_map_color_palette

    @property
    def frame_color_palette(self):
        return hm3d_9cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return None

    @property
    def num_sem_categories(self):
        return len(self.hm3d_9cats)

class HM3D_UnseenCat(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to 10 indoor
    category IDs.
    """
    def __init__(self):
        super().__init__()
        self.hm3d_cats = json.load(open('data/goat_unseen.json','r'))
        self.hm3d_cat_name_to_cat_id = {x:idx for idx,x in enumerate(self.hm3d_cats)}
        self.hm3d_goal_id_to_goal_name = {idx:x for idx,x in enumerate(self.hm3d_cats)}
        self.hm3d_goal_id_to_goal_id = {i:i for i in range(len(self.hm3d_cats))}
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_goal_id[goal_id],
            self.hm3d_goal_id_to_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, objects):
        self._instance_id_to_category_id = np.array(
            [
                self.hm3d_cat_name_to_cat_id.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip(), 'unknown'),
                    self.num_sem_categories - 1,
                )
                for obj in objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return hm3d_92cat_map_color_palette

    @property
    def frame_color_palette(self):
        return hm3d_92cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return None

    @property
    def num_sem_categories(self):
        return len(self.hm3d_cats)

class OVON_Cat(SemanticCategoryMapping):
    '''
    Semantic category mapping for OVON
    - num of categories: 379 
    - train: 280
    - val_seen: 79
    - val_seen_synonyms: 50
    - val_unseen: 49
    '''
    def __init__(self):
        super().__init__()
        self.cats = {
            "cooker": 0,
            "chair": 1,
            "window": 2,
            "couch": 3,
            "bathroom cabinet": 4,
            "sink": 5,
            "bathtub": 6,
            "towel": 7,
            "box": 8,
            "desk": 9,
            "toilet": 10,
            "cloth": 11,
            "washing machine": 12,
            "laundry basket": 13,
            "cabinet": 14,
            "blanket": 15,
            "table": 16,
            "tv": 17,
            "closet": 18,
            "bed": 19,
            "treadmill": 20,
            "shelving": 21,
            "monitor": 22,
            "photo stand": 23,
            "radio": 24,
            "lamp stand": 25,
            "wardrobe": 26,
            "bath towel": 27,
            "sink cabinet": 28,
            "closet shelf": 29,
            "kitchen top": 30,
            "oven": 31,
            "kitchen lower cabinet": 32,
            "bathroom towel": 33,
            "shower cabin": 34,
            "bathroom shelf": 35,
            "board": 36,
            "clothes": 37,
            "trashcan": 38,
            "bucket": 39,
            "bed curtain": 40,
            "sofa chair": 41,
            "display cabinet": 42,
            "copier machine": 43,
            "shower": 44,
            "bathrobe": 45,
            "stovetop": 46,
            "aquarium": 47,
            "cardboard box": 48,
            "cardboard": 49,
            "stack of papers": 50,
            "grill": 51,
            "pillar": 52,
            "lounge chair": 53,
            "seat": 54,
            "bag": 55,
            "shower tap": 56,
            "window shutter": 57,
            "oven and stove": 58,
            "worktop": 59,
            "bar": 60,
            "fire extinguisher": 61,
            "washbasin": 62,
            "candle": 63,
            "hat": 64,
            "kitchen appliance": 65,
            "washer-dryer": 66,
            "telescope": 67,
            "stand": 68,
            "basket": 69,
            "heater": 70,
            "bench": 71,
            "toy": 72,
            "pouffe": 73,
            "stove": 74,
            "rack": 75,
            "plush toy": 76,
            "container": 77,
            "clock": 78,
            "drawer": 79,
            "pool table": 80,
            "bar cabinet": 81,
            "display table": 82,
            "screen": 83,
            "hunting trophy": 84,
            "water dispenser": 85,
            "antique telephone": 86,
            "stool": 87,
            "kitchen countertop items": 88,
            "bathtub platform": 89,
            "case": 90,
            "vacuum cleaner": 91,
            "duct": 92,
            "ladder": 93,
            "globe": 94,
            "board game": 95,
            "air conditioner": 96,
            "record player": 97,
            "sideboard": 98,
            "paper": 99,
            "grass": 100,
            "sign": 101,
            "sofa seat": 102,
            "patio chair": 103,
            "shower-bath cabinet": 104,
            "bathroom accessory": 105,
            "dinner table": 106,
            "clutter": 107,
            "tool": 108,
            "countertop": 109,
            "plate": 110,
            "tray": 111,
            "range hood": 112,
            "storage shelving": 113,
            "crib": 114,
            "kitchen cabinet lower": 115,
            "tank": 116,
            "cabinet table": 117,
            "office table": 118,
            "shower soap shelf": 119,
            "support beam": 120,
            "office chair": 121,
            "bottle": 122,
            "cart": 123,
            "flag": 124,
            "umbrella": 125,
            "whiteboard": 126,
            "coat": 127,
            "urinal": 128,
            "stone support structure": 129,
            "arcade game": 130,
            "electrical controller": 131,
            "decorative quilt": 132,
            "storage cabinet": 133,
            "wood": 134,
            "clothes dryer": 135,
            "painting frame": 136,
            "locker": 137,
            "handle": 138,
            "laundry": 139,
            "pantry": 140,
            "newspaper": 141,
            "cutting board": 142,
            "crate": 143,
            "shower bar": 144,
            "pot": 145,
            "shade": 146,
            "baby changing station": 147,
            "ornament": 148,
            "curtain valence": 149,
            "desk lamp": 150,
            "handbag": 151,
            "scarf": 152,
            "kitchen extractor": 153,
            "antique clock": 154,
            "shelf cubby": 155,
            "jacuzzi": 156,
            "firewood": 157,
            "note": 158,
            "brochure": 159,
            "banner": 160,
            "drawer sink table": 161,
            "canvas": 162,
            "fuse box": 163,
            "telephone": 164,
            "sheet": 165,
            "elephant sculpture": 166,
            "rocking chair": 167,
            "jewelry box": 168,
            "candle holder": 169,
            "bed comforter": 170,
            "decorative plate": 171,
            "book rack": 172,
            "iron board": 173,
            "jacket": 174,
            "pool": 175,
            "backrest": 176,
            "easy chair": 177,
            "headboard": 178,
            "swivel chair": 179,
            "dish rack": 180,
            "cradle": 181,
            "tent": 182,
            "medical lamp": 183,
            "relief": 184,
            "massage bed": 185,
            "laptop": 186,
            "double armchair": 187,
            "foot spa": 188,
            "spa bench": 189,
            "sauna oven": 190,
            "cosmetics": 191,
            "exhibition panel": 192,
            "pile of magazines": 193,
            "solarium": 194,
            "shower rail": 195,
            "figure": 196,
            "light switch": 197,
            "balustrade": 198,
            "high shelf": 199,
            "water tank": 200,
            "elevator": 201,
            "kitchen table": 202,
            "gym equipment": 203,
            "schedule": 204,
            "wine cabinet": 205,
            "paper storage": 206,
            "kitchen shelf": 207,
            "artwork": 208,
            "closet shelving": 209,
            "window shade": 210,
            "counter": 211,
            "amplifier": 212,
            "hose": 213,
            "gas furnace": 214,
            "sleeping bag": 215,
            "trampoline": 216,
            "folding table": 217,
            "hammock": 218,
            "balcony railing": 219,
            "staircase trim": 220,
            "motorcycle": 221,
            "cleaning clutter": 222,
            "dishrag": 223,
            "sofa set": 224,
            "purse": 225,
            "washbasin counter": 226,
            "tile": 227,
            "shower stall": 228,
            "round chair": 229,
            "violin case": 230,
            "folding stand": 231,
            "wreath": 232,
            "clothes bag": 233,
            "paper towel": 234,
            "bunk bed": 235,
            "spice rack": 236,
            "cabinet clutter": 237,
            "dressing table": 238,
            "lampshade": 239,
            "bulletin board": 240,
            "mantle": 241,
            "power breaker box": 242,
            "storage": 243,
            "binder": 244,
            "dish cabinet": 245,
            "kitchen sink": 246,
            "robe": 247,
            "hutch": 248,
            "icebox": 249,
            "mixer": 250,
            "storage space": 251,
            "archway": 252,
            "sink table": 253,
            "barbecue": 254,
            "water fountain": 255,
            "sewing machine": 256,
            "magazine": 257,
            "clothing stand": 258,
            "jar": 259,
            "chimney": 260,
            "fence": 261,
            "book cabinet": 262,
            "desk cabinet": 263,
            "stack of stuff": 264,
            "foosball game table": 265,
            "stereo set": 266,
            "guitar frame": 267,
            "sunbed": 268,
            "workstation": 269,
            "hook": 270,
            "casket": 271,
            "sled": 272,
            "soft chair": 273,
            "table stand": 274,
            "backsplash": 275,
            "fish tank": 276,
            "stage": 277,
            "gate": 278,
            "exercise ladder": 279,
            "curtain": 280,
            "fireplace": 281,
            "armchair": 282,
            "lamp": 283,
            "kitchen cabinet": 284,
            "shower curtain": 285,
            "shower tub": 286,
            "bath cabinet": 287,
            "bath sink": 288,
            "cushion": 289,
            "window frame": 290,
            "lamp table": 291,
            "tv stand": 292,
            "dining table": 293,
            "dining chair": 294,
            "kitchen counter": 295,
            "desk chair": 296,
            "shelf": 297,
            "folding chair": 298,
            "railing": 299,
            "coffee table": 300,
            "electric box": 301,
            "throw blanket": 302,
            "window curtain": 303,
            "stairs railing": 304,
            "table lamp": 305,
            "bar chair": 306,
            "furnace": 307,
            "ironing board": 308,
            "chest of drawers": 309,
            "tablecloth": 310,
            "backpack": 311,
            "refrigerator cabinet": 312,
            "computer desk": 313,
            "exercise equipment": 314,
            "computer chair": 315,
            "bedside lamp": 316,
            "bed table": 317,
            "kitchen island": 318,
            "bathroom counter": 319,
            "chest drawer": 320,
            "appliance": 321,
            "bicycle": 322,
            "curtain rod": 323,
            "clothes rack": 324,
            "bed sheet": 325,
            "bookshelf": 326,
            "computer": 327,
            "file cabinet": 328,
            "water heater": 329,
            "picture": 330,
            "mirror": 331,
            "clothes hanger rod": 332,
            "nightstand": 333,
            "dresser": 334,
            "flower vase": 335,
            "microwave": 336,
            "blinds": 337,
            "refrigerator": 338,
            "stair": 339,
            "rug": 340,
            "pillow": 341,
            "dishwasher": 342,
            "freezer": 343,
            "carpet": 344,
            "boiler": 345,
            "staircase handrail": 346,
            "plant": 347,
            "radiator": 348,
            "vase": 349,
            "decorative plant": 350,
            "statue": 351,
            "photo": 352,
            "printer": 353,
            "exercise bike": 354,
            "shower glass": 355,
            "handrail": 356,
            "photo mount": 357,
            "book": 358,
            "window glass": 359,
            "parapet": 360,
            "glass": 361,
            "flowerpot": 362,
            "coffee machine": 363,
            "footstool": 364,
            "calendar": 365,
            "hanging clothes": 366,
            "island": 367,
            "speaker": 368,
            "ottoman": 369,
            "hanger": 370,
            "bedframe": 371,
            "shower dial": 372,
            "guitar": 373,
            "bowl of fruit": 374,
            "piano": 375,
            "christmas tree": 376,
            "footrest": 377,
            "food": 378
        }

        self._categories = list(self.cats.keys()) + ['unknown']
        self._category_name_to_category_id = {x: idx for idx, x in enumerate(self._categories)}
        self._category_id_to_category_name = {idx: x for idx, x in enumerate(self._categories)}
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        category_id = goal_id
        return (
            category_id, # goal category id
            self._category_id_to_category_name[category_id], # goal category id
        )
    
    def reset_instance_id_to_category_id(self, objects):        
        self._instance_id_to_category_id = []

        instance_names = [obj.category.name().lower().strip() for obj in objects]
        for instance_name in instance_names:
            _id = self._category_name_to_category_id.get(instance_name, self.num_sem_categories - 1)
            if _id == self.num_sem_categories - 1:
                translated_instance_name = hm3d_category_name_to_mp3d_category_name.get(instance_name, 'unknown')
                _id = self._category_name_to_category_id.get(translated_instance_name, self.num_sem_categories - 1)
            self._instance_id_to_category_id.append(_id)

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return hm3d_9cat_map_color_palette

    @property
    def frame_color_palette(self):
        return hm3d_9cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return None

    @property
    def num_sem_categories(self):
        return len(self._categories)

    @property
    def num_categories(self):
        return len(self._categories)

class PaletteIndices:
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    BEEN_CLOSE = 6
    SHORT_TERM_GOAL = 7
    INSTANCE_BORDER = 8
    SEM_START = 9

from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal


# ============================================================
# Oracle Success
# ============================================================

@registry.register_measure
class OracleSuccess(Measure):
    """
    Oracle Success (OS)

    只要 agent 在整条轨迹中的任意时刻，
    distance_to_goal < success_distance，
    则本 episode 的 oracle_success = 1。

    与普通 Success 不同：
    不要求执行 STOP。
    """

    cls_uuid: str = "oracle_success"

    def __init__(
        self,
        sim: Simulator,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._success_distance = config.success_distance

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        # OracleSuccess 依赖 distance_to_goal
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DistanceToGoal.cls_uuid],
        )

        self._metric = 0.0

        self.update_metric(
            episode=episode,
            task=task,
            *args,
            **kwargs,
        )

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        # 一旦成功，之后一直保持 1
        if distance_to_target < self._success_distance:
            self._metric = 1.0


# ============================================================
# Hydra config
# ============================================================

@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 0.25


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.measurements.oracle_success",
    group="habitat/task/measurements",
    name="oracle_success",
    node=OracleSuccessMeasurementConfig,
)
"""Task registry for release-time navigation evaluation.

This package hosts task-specific ``VLNEvaluator`` implementations. The unified
entrypoint ``evaluate_navigation.py`` dispatches to the correct evaluator via
:func:`get_evaluator`.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Type

# Resolve bundled evaluation utilities when this package is imported directly.
_EVALUATION_ROOT = str(Path(__file__).resolve().parents[1])
if _EVALUATION_ROOT not in sys.path:
    sys.path.insert(0, _EVALUATION_ROOT)

__all__ = ["get_evaluator"]

# Task-name buckets used for dispatching. Names mirror the values accepted by
# the ``--task`` CLI flag so that the three ``srun_eval_*.sh`` launchers work
# unchanged.
_OBJECT_GOAL_TASKS = {"objectnav", "psl", "ovon", "object"}
_IIGN_TASKS = {"instance_dialog", "coin", "instance"}

if TYPE_CHECKING:  # pragma: no cover - typing only
    # ``VLNEvaluator`` is defined identically-named in both task modules; the
    # ``TYPE_CHECKING`` guard avoids importing heavy dependencies here.
    from .objectnav import VLNEvaluator as _ObjectGoalEvaluator  # noqa: F401
    from .iign import VLNEvaluator as _IIGNEvaluator  # noqa: F401


def get_evaluator(task: str, dialog_enabled: bool = False) -> Type:
    """Return the ``VLNEvaluator`` class matching ``task``.

    Parameters
    ----------
    task:
        Value coming from ``--task`` on the CLI.
    dialog_enabled:
        When ``True`` we always route to the IIGN evaluator; CoIN /
        instance_dialog launchers use this flag to force that branch.

    Raises
    ------
    ValueError
        If ``task`` does not match any known bucket.
    """

    task_key = (task or "").strip().lower()

    # Explicit dialog flag wins over task-name heuristics because CoIN /
    # instance_dialog runs used to force the IIGN branch this way.
    if dialog_enabled or task_key in _IIGN_TASKS:
        # Lazy import so that Object Goal only environments never touch the
        # IIGN-specific ``LanguageUncertaintyDatasetV1`` dependency.
        from .iign import VLNEvaluator as IIGNEvaluator
        return IIGNEvaluator

    if task_key in _OBJECT_GOAL_TASKS:
        from .objectnav import VLNEvaluator as ObjectGoalEvaluator
        return ObjectGoalEvaluator

    supported = sorted(_OBJECT_GOAL_TASKS | _IIGN_TASKS)
    raise ValueError(
        f"Unknown --task value: {task!r}. Supported tasks are: {supported}. "
        "Pass --dialog_enabled to force the IIGN evaluator regardless of task name."
    )

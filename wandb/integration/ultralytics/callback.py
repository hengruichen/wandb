import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union

from packaging import version

try:
    import dill as pickle
except ImportError:
    import pickle

import wandb
from wandb.sdk.lib import telemetry

try:
    import torch
    import ultralytics
    from tqdm.auto import tqdm

    if version.parse(ultralytics.__version__) > version.parse("8.0.186"):
        wandb.termwarn(
            """This integration is tested and supported for ultralytics v8.0.186 and below.
            Please report any issues to https://github.com/wandb/wandb/issues with the tag `yolov8`.""",
            repeat=False,
        )

    from ultralytics.models import YOLO
    from ultralytics.models.yolo.classify import (
        ClassificationPredictor,
        ClassificationTrainer,
        ClassificationValidator,
    )
    from ultralytics.models.yolo.detect import (
        DetectionPredictor,
        DetectionTrainer,
        DetectionValidator,
    )
    from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator
    from ultralytics.models.yolo.segment import (
        SegmentationPredictor,
        SegmentationTrainer,
        SegmentationValidator,
    )
    from ultralytics.utils.torch_utils import de_parallel
    try:
        from ultralytics.yolo.utils import RANK, __version__
    except ModuleNotFoundError:
        from ultralytics.utils import RANK, __version__

    from wandb.integration.ultralytics.bbox_utils import (
        plot_predictions,
        plot_validation_results,
    )
    from wandb.integration.ultralytics.classification_utils import (
        plot_classification_predictions,
        plot_classification_validation_results,
    )
    from wandb.integration.ultralytics.mask_utils import (
        plot_mask_predictions,
        plot_mask_validation_results,
    )
    from wandb.integration.ultralytics.pose_utils import (
        plot_pose_predictions,
        plot_pose_validation_results,
    )
except ImportError as e:
    wandb.Error(e)


TRAINER_TYPE = Union[
    ClassificationTrainer, DetectionTrainer, SegmentationTrainer, PoseTrainer
]
VALIDATOR_TYPE = Union[
    ClassificationValidator, DetectionValidator, SegmentationValidator, PoseValidator
]
PREDICTOR_TYPE = Union[
    ClassificationPredictor, DetectionPredictor, SegmentationPredictor, PosePredictor
]


class WandBUltralyticsCallback:
    """Stateful callback for logging to W&B.

    In particular, it will log model checkpoints, predictions, and
    ground-truth annotations with interactive overlays for bounding boxes
    to Weights & Biases Tables during training,
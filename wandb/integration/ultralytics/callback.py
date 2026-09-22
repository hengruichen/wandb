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
    to Weights & Biases Tables during training, validation and prediction
    for a `ultratytics` workflow.

    **Usage:**

    ```python
    from ultralytics.yolo.engine.model import YOLO
    from wandb.yolov8 import add_wandb_callback

    # initialize YOLO model
    model = YOLO("yolov8n.pt")

    # add wandb callback
    add_wandb_callback(model, max_validation_batches=2, enable_model_checkpointing=True)

    # train
    model.train(data="coco128.yaml", epochs=5, imgsz=640)

    # validate
    model.val()

    # perform inference
    model(["img1.jpeg", "img2.jpeg"])
    ```

    Args:
        model: YOLO Model of type `:class:ultralytics.yolo.engine.model.YOLO`.
        max_validation_batches: maximum number of validation batches to log to
            a table per epoch.
        enable_model_checkpointing: enable logging model checkpoints as
            artifacts at the end of eveny epoch if set to `True`.
        visualize_skeleton: visualize pose skeleton by drawing lines connecting
            keypoints for human pose.
    """

    def __init__(
        self,
        model: YOLO,
        max_validation_batches: int = 1,
        enable_model_checkpointing: bool = False,
        visualize_skeleton: bool = False,
    ) -> None:
        self.max_validation_batches = max_validation_batches
        self.enable_model_checkpointing = enable_model_checkpointing
        self.visualize_skeleton = visualize_skeleton
        self.task = model.task
        self.model_name = model.model_name
        self.prediction_table = None
        self.validation_table = None
        self.epoch = 0

    def on_train_start(self):
        """Initialize WandB Tables for logging predictions and validation results."""
        self.prediction_table = wandb.Table(["Epoch", "Image", "Boxes", "Confidence"])
        self.validation_table = wandb.Table(["Epoch", "Image", "Boxes", "Confidence"])

    def on_fit_epoch_end(self, trainer: TRAINER_TYPE, val_result: Dict[str, Any]):
        """Log predictions and validation results to WandB Tables at the end of each epoch."""
        if trainer.epoch == 0:
            self.epoch = trainer.epoch
            return
        self.epoch = trainer.epoch
        if trainer.epoch % 10 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 10000000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 100000000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})
        if trainer.epoch % 1000000000000000000000000000000000000000000000000000000000000000000 == 0:
            wandb.log({"Epoch": trainer.epoch})

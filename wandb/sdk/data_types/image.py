import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse

import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath

from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib  # type: ignore
    import numpy as np
    import torch  # type: ignore
    from PIL.Image import Image as PILImage

    from wandb.sdk.artifacts.artifact import Artifact

    from ..wandb_run import Run as LocalRun

    ImageDataType = Union[
        "matplotlib.artist.Artist", "PILImage", "TorchTensorType", "np.ndarray"
    ]
    ImageDataOrPathType = Union[str, "Image", ImageDataType]
    TorchTensorType = Union["torch.Tensor", "torch.Variable"]


def _server_accepts_image_filenames() -> bool:
    if util._is_offline():
        return True

    # Newer versions of wandb accept large image filenames arrays
    # but older versions would have issues with this.
    max_cli_version = util._get_max_cli_version()
    if max_cli_version is None:
        return False
    from pkg_resources import parse_version

    accepts_image_filenames: bool = parse_version("0.12.10") <= parse_version(
        max_cli_version
    )
    return accepts_image_filenames


def _server_accepts_artifact_path() -> bool:
    from pkg_resources import parse_version

    target_version = "0.12.14"
    max_cli_version = util._get_max_cli_version() if not util._is_offline() else None
    accepts_artifact_path: bool = max_cli_version is not None and parse_version(
        target_version
    ) <= parse_version(max_cli_version)
    return accepts_artifact_path


class Image(BatchableMedia):
    """Format images for logging to W&B.

    Arguments:
        data_or_path: (numpy array, string, io) Accepts numpy array of
            image data, or a PIL image. The class attempts to infer
            the data format and converts it.
        mode: (string) The PIL mode for an image. Most common are "L", "RGB",
            "RGBA". Full explanation at https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes.
        caption: (string) Label for display of image.

    Note : When logging a `torch.Tensor` as a `wandb.Image`, images are normalized. If you do not want to normalize your images, please convert your tensors to a PIL Image.

    Examples:
        ### Create a wandb.Image from a numpy array
        <!--yeadoc-test:log-image-numpy-->
        ```python
        import numpy as np
        import wandb

        wandb.init()
        examples = []
        for i in range(3):
            pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
            image = wandb.Image(pixels, caption=f"random field {i}")
            examples.append(image)
        wandb.log({"examples": examples})
        ```

        ### Create a wandb.Image from a PILImage
        <!--yeadoc-test:log-image-pillow-->
        ```python
        import numpy as np
        from PIL import Image as PILImage
        import wandb

        wandb.init()
        examples = []
        for i in range(3):
            pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
            pil_image = PILImage.fromarray(pixels, mode="RGB")
            image = wandb.Image(pil_image, caption=f"random field {i}")
            examples.append(image)
        wandb.log({"examples": examples})
        ```
    """

    MAX_ITEMS = 108

    # PIL limit
    MAX_DIMENSION = 65500

    _log_type = "image-file"

    format: Optional[str]
    _grouping: Optional[int]
    _caption: Optional[str]
    _width: Optional[int]
    _height: Optional[int]
    _image: Optional["PILImage"]
    _classes: Optional["Cla
# ... [truncated] ...
           )

        captions = Image.all_captions(seq)

        if captions:
            meta["captions"] = captions

        all_masks = Image.all_masks(seq, run, key, step)

        if all_masks:
            meta["all_masks"] = all_masks

        all_boxes = Image.all_boxes(seq, run, key, step)

        if all_boxes:
            meta["all_boxes"] = all_boxes

        return meta

    @classmethod
    def all_masks(
        cls: Type["Image"],
        images: Sequence["Image"],
        run: "LocalRun",
        run_key: str,
        step: Union[int, str],
    ) -> Union[List[Optional[dict]], bool]:
        all_mask_groups: List[Optional[dict]] = []
        for image in images:
            if image._masks:
                mask_group = {}
                for k in image._masks:
                    mask = image._masks[k]
                    mask_group[k] = mask.to_json(run)
                all_mask_groups.append(mask_group)
            else:
                all_mask_groups.append(None)
        if all_mask_groups and not all(x is None for x in all_mask_groups):
            return all_mask_groups
        else:
            return False

    @classmethod
    def all_boxes(
        cls: Type["Image"],
        images: Sequence["Image"],
        run: "LocalRun",
        run_key: str,
        step: Union[int, str],
    ) -> Union[List[Optional[dict]], bool]:
        all_box_groups: List[Optional[dict]] = []
        for image in images:
            if image._boxes:
                box_group = {}
                for k in image._boxes:
                    box = image._boxes[k]
                    box_group[k] = box.to_json(run)
                all_box_groups.append(box_group)
            else:
                all_box_groups.append(None)
        if all_box_groups and not all(x is None for x in all_box_groups):
            return all_box_groups
        else:
            return False

    @classmethod
    def all_captions(
        cls: Type["Image"], images: Sequence["Media"]
    ) -> Union[bool, Sequence[Optional[str]]]:
        return cls.captions(images)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return False
        else:
            if self.path_is_reference(self._path) and self.path_is_reference(
                other._path
            ):
                return self._path == other._path
            self_image = self.image
            other_image = other.image
            if self_image is not None:
                self_image = list(self_image.getdata())  # type: ignore
            if other_image is not None:
                other_image = list(other_image.getdata())  # type: ignore

            return (
                self._grouping == other._grouping
                and self._caption == other._caption
                and self._width == other._width
                and self._height == other._height
                and self_image == other_image
                and self._classes == other._classes
            )

    def to_data_array(self) -> List[Any]:
        res = []
        if self.image is not None:
            data = list(self.image.getdata())
            for i in range(self.image.height):
                res.append(data[i * self.image.width : (i + 1) * self.image.width])
        self._free_ram()
        return res

    def _free_ram(self) -> None:
        if self._path is not None:
            self._image = None

    @property
    def image(self) -> Optional["PILImage"]:
        if self._image is None:
            if self._path is not None and not self.path_is_reference(self._path):
                pil_image = util.get_module(
                    "PIL.Image",
                    required='wandb.Image needs the PIL package. To get it, run "pip install pillow".',
                )
                self._image = pil_image.open(self._path)
                self._image.load()
        return self._image


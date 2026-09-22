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
    _classes: Optional["Classes"]

    def __init__(
        self,
        data_or_path: ImageDataOrPathType,
        *,
        caption: Optional[str] = None,
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        classes: Optional[Classes] = None,
    ) -> None:
        self._path = None
        self._grouping = None
        self._caption = None
        self._width = None
        self._height = None
        self._classes = None
        self._masks = {}
        self._boxes = {}
        self._image = None

        if isinstance(data_or_path, str):
            self._path = data_or_path
        elif isinstance(data_or_path, Image):
            self._grouping = data_or_path._grouping
            self._caption = data_or_path._caption
            self._width = data_or_path._width
            self._height = data_or_path._height
            self._classes = data_or_path._classes
            self._masks = data_or_path._masks
            self._boxes = data_or_path._boxes
            self._image = data_or_path._image
        elif isinstance(data_or_path, PILImage):
            self._image = data_or_path
            self._caption = caption
            self._width = width
            self._height = height
            self._classes = classes
        elif isinstance(data_or_path, np.ndarray):
            self._image = wandb.utils.convert_numpy_image_to_pil_image(
                data_or_path, format
            )
            self._caption = caption
            self._width = width
            self._height = height
            self._classes = classes
        elif isinstance(data_or_path, torch.Tensor):
            self._image = wandb.utils.convert_tensor_to_pil_image(
                data_or_path, format
            )
            self._caption = caption
            self._width = width
            self._height = height
            self._classes = classes
        elif isinstance(data_or_path, matplotlib.artist.Artist):
            self._image = wandb.utils.convert_artist_to_pil_image(
                data_or_path, format
            )
            self._caption = caption
            self._width = width
            self._height = height
            self._classes = classes
        else:
            raise ValueError(
                f"Unsupported data type for wandb.Image: {type(data_or_path)}"
            )

        if self._image is not None:
            self._width = self._image.width
            self._height = self._image.height
            if format is None:
                self._format = self._image.mode
            else:
                self._format = format

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def grouping(self) -> Optional[int]:
        return self._grouping

    @property
    def caption(self) -> Optional[str]:
        return self._caption

    @property
    def width(self) -> Optional[int]:
        return self._width

    @property
    def height(self) -> Optional[int]:
        return self._height

    @property
    def classes(self) -> Optional[Classes]:
        return self._classes

    @property
    def masks(self) -> Dict[str, ImageMask]:
        return self._masks

    @property
    def boxes(self) -> Dict[str, BoundingBoxes2D]:
        return self._boxes

    @property
    def image(self) -> Optional["PILImage"]:
        return self._image

    def to_json(self, run: "LocalRun") -> Dict[str, Any]:
        if self._path is not None and not self.path_is_reference(self._path):
            return {"path": self._path, "type": "image-file"}
        else:
            meta = {}
            if self._grouping is not None:
                meta["grouping"] = self._grouping
            if self._caption is not None:
                meta["caption"] = self._caption
            if self._width is not None:
                meta["width"] = self._width
            if self._height is not None:
                meta["height"] = self._height
            if self._classes is not None:
                meta["classes"] = self._classes.to_json(run)
            if self._masks:
                masks = {}
                for k in self._masks:
                    masks[k] = self._masks[k].to_json(run)
                meta["masks"] = masks
            if self._boxes:
                boxes = {}
                for k in self._boxes:
                    boxes[k] = self._boxes[k].to_json(run)
                meta["boxes"] = boxes
            return {"data": self.to_data_array(), "meta": meta}

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

    @classmethod
    def from_artifact(
        cls: Type["Image"],
        artifact: "Artifact",
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from an artifact.

        Arguments:
            artifact: (Artifact) The artifact to create the image from.
            key: (string) The key of the image in the artifact.
            step: (int or string) The step of the image in the artifact.

        Returns:
            wandb.Image: The image from the artifact.
        """
        if not _server_accepts_artifact_path():
            raise ValueError(
                "The server does not support logging images from artifacts."
            )

        if key is None:
            key = "image"

        if step is None:
            step = "latest"

        path = f"{artifact.full_path}/{key}/{step}"

        return cls(path=path)

    @classmethod
    def from_reference(
        cls: Type["Image"],
        reference: "LogicalPath",
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference.

        Arguments:
            reference: (wandb.sdk.lib.paths.LogicalPath) The reference to create the image from.
            key: (string) The key of the image in the reference.
            step: (int or string) The step of the image in the reference.

        Returns:
            wandb.Image: The image from the reference.
        """
        if key is None:
            key = "image"

        if step is None:
            step = "latest"

        path = f"{reference}/{key}/{step}"

        return cls(path=path)

    @classmethod
    def from_url(
        cls: Type["Image"],
        url: str,
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a URL.

        Arguments:
            url: (string) The URL to create the image from.
            key: (string) The key of the image in the URL.
            step: (int or string) The step of the image in the URL.

        Returns:
            wandb.Image: The image from the URL.
        """
        if not _server_accepts_image_filenames():
            raise ValueError(
                "The server does not support logging images from URLs."
            )

        if key is None:
            key = "image"

        if step is None:
            step = "latest"

        path = f"{url}/{key}/{step}"

        return cls(path=path)

    @classmethod
    def from_reference_or_url(
        cls: Type["Image"],
        reference_or_url: Union["LogicalPath", str],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference or URL.

        Arguments:
            reference_or_url: (wandb.sdk.lib.paths.LogicalPath or string) The reference or URL to create the image from.
            key: (string) The key of the image in the reference or URL.
            step: (int or string) The step of the image in the reference or URL.

        Returns:
            wandb.Image: The image from the reference or URL.
        """
        if isinstance(reference_or_url, str):
            return cls.from_url(
                url=reference_or_url, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_url, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact(
        cls: Type["Image"],
        reference_or_artifact: Union["Artifact", "LogicalPath"],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference or artifact.

        Arguments:
            reference_or_artifact: (wandb.sdk.lib.paths.LogicalPath or wandb.sdk.artifacts.artifact.Artifact) The reference or artifact to create the image from.
            key: (string) The key of the image in the reference or artifact.
            step: (int or string) The step of the image in the reference or artifact.

        Returns:
            wandb.Image: The image from the reference or artifact.
        """
        if isinstance(reference_or_artifact, str):
            return cls.from_url(
                url=reference_or_artifact, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_artifact(
                artifact=reference_or_artifact, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url(
        cls: Type["Image"],
        reference_or_artifact_or_url: Union[
            "Artifact", "LogicalPath", str
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, or URL.

        Arguments:
            reference_or_artifact_or_url: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, or string) The reference, artifact, or URL to create the image from.
            key: (string) The key of the image in the reference, artifact, or URL.
            step: (int or string) The step of the image in the reference, artifact, or URL.

        Returns:
            wandb.Image: The image from the reference, artifact, or URL.
        """
        if isinstance(reference_or_artifact_or_url, str):
            return cls.from_url(
                url=reference_or_artifact_or_url, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image: Union[
            "Artifact", "Image", "LogicalPath", str
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, or image.

        Arguments:
            reference_or_artifact_or_url_or_image: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, or image to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, or image.
            step: (int or string) The step of the image in the reference, artifact, URL, or image.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, or image.
        """
        if isinstance(reference_or_artifact_or_url_or_image, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image, Image):
            return reference_or_artifact_or_url_or_image
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path: Union[
            "Artifact", "Image", "LogicalPath", str
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, or path.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, or path to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, or path.
            step: (int or string) The step of the image in the reference, artifact, URL, image, or path.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, or path.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path, Image):
            return reference_or_artifact_or_url_or_image_or_path
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict: Union[
            "Artifact", "Image", "LogicalPath", str, Dict[str, Any]
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, path, or dictionary.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path_or_dict: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, path, or dictionary to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, path, or dictionary.
            step: (int or string) The step of the image in the reference, artifact, URL, image, path, or dictionary.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, path, or dictionary.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path_or_dict, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path_or_dict, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict, Image):
            return reference_or_artifact_or_url_or_image_or_path_or_dict
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict, LogicalPath):
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict, Dict):
            return cls.from_dict(
                dict=reference_or_artifact_or_url_or_image_or_path_or_dict, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict_or_list(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict_or_list: Union[
            "Artifact", "Image", "LogicalPath", str, Dict[str, Any], List[Any]
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, path, dictionary, or list.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path_or_dict_or_list: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, path, dictionary, or list to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, path, dictionary, or list.
            step: (int or string) The step of the image in the reference, artifact, URL, image, path, dictionary, or list.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, path, dictionary, or list.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, Image):
            return reference_or_artifact_or_url_or_image_or_path_or_dict_or_list
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, LogicalPath):
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, Dict):
            return cls.from_dict(
                dict=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, List):
            return cls.from_list(
                list=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple: Union[
            "Artifact", "Image", "LogicalPath", str, Dict[str, Any], List[Any], Tuple[Any, ...]
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, path, dictionary, list, or tuple.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, path, dictionary, list, or tuple to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, path, dictionary, list, or tuple.
            step: (int or string) The step of the image in the reference, artifact, URL, image, path, dictionary, list, or tuple.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, path, dictionary, list, or tuple.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, Image):
            return reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, LogicalPath):
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, Dict):
            return cls.from_dict(
                dict=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, List):
            return cls.from_list(
                list=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, Tuple):
            return cls.from_tuple(
                tuple=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set: Union[
            "Artifact", "Image", "LogicalPath", str, Dict[str, Any], List[Any], Tuple[Any, ...], Set[Any]
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, path, dictionary, list, tuple, or set.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, path, dictionary, list, tuple, or set to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, path, dictionary, list, tuple, or set.
            step: (int or string) The step of the image in the reference, artifact, URL, image, path, dictionary, list, tuple, or set.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, path, dictionary, list, tuple, or set.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, Image):
            return reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, LogicalPath):
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, Dict):
            return cls.from_dict(
                dict=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, List):
            return cls.from_list(
                list=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, Tuple):
            return cls.from_tuple(
                tuple=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, Set):
            return cls.from_set(
                set=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator: Union[
            "Artifact", "Image", "LogicalPath", str, Dict[str, Any], List[Any], Tuple[Any, ...], Set[Any], Generator[Any, None, None]
        ],
        *,
        key: Optional[str] = None,
        step: Optional[Union[int, str]] = None,
    ) -> "Image":
        """Create a wandb.Image from a reference, artifact, URL, image, path, dictionary, list, tuple, set, or generator.

        Arguments:
            reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator: (wandb.sdk.lib.paths.LogicalPath, wandb.sdk.artifacts.artifact.Artifact, wandb.Image, or string) The reference, artifact, URL, image, path, dictionary, list, tuple, set, or generator to create the image from.
            key: (string) The key of the image in the reference, artifact, URL, image, path, dictionary, list, tuple, set, or generator.
            step: (int or string) The step of the image in the reference, artifact, URL, image, path, dictionary, list, tuple, set, or generator.

        Returns:
            wandb.Image: The image from the reference, artifact, URL, image, path, dictionary, list, tuple, set, or generator.
        """
        if isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, str):
            return cls.from_url(
                url=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Artifact):
            return cls.from_artifact(
                artifact=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Image):
            return reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, LogicalPath):
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Dict):
            return cls.from_dict(
                dict=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, List):
            return cls.from_list(
                list=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Tuple):
            return cls.from_tuple(
                tuple=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Set):
            return cls.from_set(
                set=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        elif isinstance(reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, Generator):
            return cls.from_generator(
                generator=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore
        else:
            return cls.from_reference(
                reference=reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator, key=key, step=step
            )  # type: ignore

    @classmethod
    def from_reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator_or_iterable(
        cls: Type["Image"],
        reference_or_artifact_or_url_or_image_or_path_or_dict_or_list_or_tuple_or_set_or_generator_or_iterable: Union[
           
"""OpenArm policy transforms for dual-arm robot with 3 cameras and 16-dim actions."""

import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_openarm_example() -> dict:
    """Creates a random input example for the OpenArm policy."""
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up and fold the clothes on the table",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to standard HWC uint8 format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Transform OpenArm data to model input format.
    
    Handles 3 cameras (top, left, right) and 16-dim state for dual-arm robot.
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/base_0_rgb"])
        left_wrist = _parse_image(data["observation/left_wrist_0_rgb"])
        right_wrist = _parse_image(data["observation/right_wrist_0_rgb"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        
        inputs["prompt"] = "Tidy up the table."

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Extract 16-dim actions for dual-arm robot."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}

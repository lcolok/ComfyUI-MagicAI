import numpy as np
import torch
from typing import Union, Tuple, Optional
from magicai.utils.converters import novel_numpy2tensor, novel_tensor2numpy


def get_shape_type(masks: Union[np.ndarray, torch.Tensor]) -> str:
    """
    Determine the shape type of the input mask.

    Args:
        masks (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.

    Returns:
        str: Shape type of the input mask.
    """
    if not isinstance(masks, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")

    ndim = masks.ndim
    shape = masks.shape

    if ndim == 4:
        if shape[1] <= 4 and shape[2] > 4 and shape[3] > 4:
            return "NCHW"
        elif shape[3] <= 4 and shape[1] > 4 and shape[2] > 4:
            return "NHWC"
    elif ndim == 3:
        if shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            return "CHW"
        elif shape[2] <= 4 and shape[0] > 4 and shape[1] > 4:
            return "HWC"
        else:
            return "NHW"
    elif ndim == 2:
        return "HW"

    return "Unknown"


def get_array_type(arr: Union[np.ndarray, torch.Tensor]) -> str:
    """
    Determine the type of the input array.

    Args:
        arr (Union[np.ndarray, torch.Tensor]): Input array or tensor.

    Returns:
        str: 'numpy' for NumPy arrays, 'torch' for PyTorch tensors.
    """
    if isinstance(arr, np.ndarray):
        return "numpy"
    elif isinstance(arr, torch.Tensor):
        return "torch"
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")


def ensure_nhwc_mask_auto(
    masks: Union[np.ndarray, torch.Tensor]
) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Automatically convert the input mask to NHWC format, maintaining the input type.

    Args:
        masks (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.

    Returns:
        Optional[Union[np.ndarray, torch.Tensor]]: Converted mask in NHWC format, or None if conversion fails.
    """
    input_type = get_array_type(masks)

    if input_type == "numpy":
        result = ensure_nhwc_mask_numpy(masks)
        return result
    elif input_type == "torch":
        result = ensure_nhwc_mask_torch(masks)
        return result


def ensure_nhwc_mask_torch(
    masks: Union[np.ndarray, torch.Tensor]
) -> Optional[torch.Tensor]:
    """
    Convert input to PyTorch tensor and then to NHWC format.

    Args:
        masks (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.

    Returns:
        Optional[torch.Tensor]: Converted tensor in NHWC format, or None if conversion fails.
    """
    if isinstance(masks, np.ndarray):
        masks = novel_numpy2tensor(masks)

    if masks.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    shape_type = get_shape_type(masks)

    conversion_map = {
        "NCHW": lambda x: x.permute(0, 2, 3, 1),
        "CHW": lambda x: x.unsqueeze(0).permute(0, 2, 3, 1),
        "HWC": lambda x: x.unsqueeze(0),
        "NHW": lambda x: x.unsqueeze(-1),
        "HW": lambda x: x.unsqueeze(0).unsqueeze(-1),
        "NHWC": lambda x: x,
    }

    return conversion_map.get(shape_type, lambda x: None)(masks)


def ensure_nhwc_mask_numpy(
    masks: Union[np.ndarray, torch.Tensor]
) -> Optional[np.ndarray]:
    """
    Convert input to NumPy array and then to NHWC format.

    Args:
        masks (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.

    Returns:
        Optional[np.ndarray]: Converted array in NHWC format, or None if conversion fails.
    """
    if isinstance(masks, torch.Tensor):
        masks = novel_tensor2numpy(masks)

    if masks.ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions.")

    shape_type = get_shape_type(masks)

    conversion_map = {
        "NCHW": lambda x: np.transpose(x, (0, 2, 3, 1)),
        "CHW": lambda x: np.transpose(x[np.newaxis, ...], (0, 2, 3, 1)),
        "HWC": lambda x: x[np.newaxis, ...],
        "NHW": lambda x: x[..., np.newaxis],
        "HW": lambda x: x[np.newaxis, ..., np.newaxis],
        "NHWC": lambda x: x,
    }

    return conversion_map.get(shape_type, lambda x: None)(masks)


class UniversalMaskConverterUtils:
    @staticmethod
    def convert_mask(
        mask: Union[np.ndarray, torch.Tensor], target_shape: str
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert mask to the target shape.

        Args:
            mask (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.
            target_shape (str): Target shape to convert to.

        Returns:
            Union[np.ndarray, torch.Tensor]: Converted mask.
        """
        target_shape = target_shape.upper()
        mask = ensure_nhwc_mask_auto(mask)

        if mask is None:
            raise ValueError("Failed to convert input mask to NHWC format.")

        if isinstance(mask, np.ndarray):
            convert_func = UniversalMaskConverterUtils.transpose
            squeeze_func = np.squeeze
        elif torch.is_tensor(mask):
            convert_func = UniversalMaskConverterUtils.permute
            squeeze_func = torch.squeeze
        else:
            raise TypeError("Input mask must be a NumPy array or PyTorch tensor.")

        return UniversalMaskConverterUtils.convert(
            mask, target_shape, convert_func, squeeze_func
        )

    @staticmethod
    def convert(
        mask: Union[np.ndarray, torch.Tensor],
        target_shape: str,
        convert_func: callable,
        squeeze_func: callable,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert mask to the target shape using provided conversion functions.

        Args:
            mask (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.
            target_shape (str): Target shape to convert to.
            convert_func (callable): Function to use for dimension conversion.
            squeeze_func (callable): Function to use for dimension reduction.

        Returns:
            Union[np.ndarray, torch.Tensor]: Converted mask.
        """
        conversion_map = {
            "NCHW": lambda x: convert_func(x, (0, 3, 1, 2)),
            "CNHW": lambda x: convert_func(x, (3, 0, 1, 2)),
            "HWC": lambda x: squeeze_func(x, axis=0),
            "CHW": lambda x: convert_func(squeeze_func(x, axis=0), (2, 0, 1)),
            "NHWC": lambda x: x,
            "NHW": lambda x: squeeze_func(x, axis=-1),
            "HW": lambda x: squeeze_func(squeeze_func(x, axis=0), axis=-1),
        }

        if target_shape not in conversion_map:
            raise ValueError(f"Unsupported target shape: {target_shape}")

        return conversion_map[target_shape](mask)

    @staticmethod
    def transpose(mask: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
        """Transpose NumPy array."""
        return mask.transpose(*axes)

    @staticmethod
    def permute(mask: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
        """Permute PyTorch tensor."""
        return mask.permute(*axes)


def get_mask_stats(mask: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Get statistics of the input mask.

    Args:
        mask (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.

    Returns:
        dict: Dictionary containing mask statistics.
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    stats = {
        "shape": mask.shape,
        "dtype": str(mask.dtype),
        "min": float(np.min(mask)),
        "max": float(np.max(mask)),
        "mean": float(np.mean(mask)),
        "std": float(np.std(mask)),
        "unique_values": np.unique(mask).tolist(),
    }

    return stats


def is_binary_mask(
    mask: Union[np.ndarray, torch.Tensor], tolerance: float = 1e-6
) -> bool:
    """
    Check if the input mask is binary.

    Args:
        mask (Union[np.ndarray, torch.Tensor]): Input mask array or tensor.
        tolerance (float): Tolerance for floating point comparison.

    Returns:
        bool: True if the mask is binary, False otherwise.
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    unique_values = np.unique(mask)
    return len(unique_values) <= 2 and np.allclose(
        unique_values, np.round(unique_values), atol=tolerance
    )

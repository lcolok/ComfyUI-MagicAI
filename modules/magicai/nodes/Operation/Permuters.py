from magicai.utils.uniformers import UniversalMaskConverterUtils
import numpy as np
import torch


class UniversalMaskConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # 假设mask以图像形式传入
                "target_shape": (["NCHW", "CNHW", "NHWC", "HWC", "CHW", "HW"],),
            },
        }

    RETURN_TYPES = ("MASK",)

    CATEGORY = "LogicAI/Operation"

    FUNCTION = "doit"

    def doit(self, mask, target_shape):
        result = UniversalMaskConverterUtils.convert_mask(mask, target_shape)
        return (result,)


class MaskToCPU:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # 假设mask以图像形式传入
            },
        }

    RETURN_TYPES = ("MASK",)

    CATEGORY = "LogicAI/Operation"

    FUNCTION = "doit"

    def doit(self, mask):
        """
        将输入的mask张量转换到CPU上。

        参数:
        mask (torch.Tensor): 输入的mask，可以位于任何设备上。

        返回:
        tuple: 包含转换到CPU上的mask张量。
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.clone().cpu()  # 克隆并转换到CPU上
        return (mask,)

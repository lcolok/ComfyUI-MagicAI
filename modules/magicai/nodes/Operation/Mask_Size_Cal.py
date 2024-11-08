import torch
import math
from torchvision.ops import masks_to_boxes
from magicai.utils.uniformers import ensure_nhwc_mask_auto


class MaskSizeCalculator:
    """
    基于mask计算合适的宽高比的节点。
    提供两种模式：
    1. area_based: 基于期望面积和mask比例计算新的宽高，确保面积不超过预期，且宽高都是8的倍数
    2. max_constrained: 在不超过期望宽高的情况下，保持mask比例计算新的宽高，同样确保是8的倍数

    输入:
        mask: NHWC 格式的mask
        expected_width: 期望宽度
        expected_height: 期望高度
        mode: 计算模式 ('area_based' 或 'max_constrained')

    输出:
        width: 计算得到的宽度 (int，被8整除)
        height: 计算得到的高度 (int，被8整除)
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expected_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "expected_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "mode": (["area_based", "max_constrained"],),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "LogicAI/Operation"

    def floor_to_multiple_of_8(self, value):
        """
        将数值向下取整到8的倍数
        """
        return math.floor(value / 8) * 8

    def calculate_dimensions(self, ratio, expected_width, expected_height, mode):
        """
        根据不同模式计算新的宽高

        Args:
            ratio: 宽高比 (width/height)
            expected_width: 期望宽度
            expected_height: 期望高度
            mode: 计算模式

        Returns:
            tuple: (width, height) 都是8的倍数
        """
        if mode == "area_based":
            # 计算期望面积
            target_area = expected_width * expected_height

            # 基于面积和比例计算新的宽高
            # height * (height * ratio) = target_area
            # height^2 * ratio = target_area
            # height = sqrt(target_area / ratio)
            height = math.sqrt(target_area / ratio)
            width = height * ratio

            # 向下取整到8的倍数
            width = self.floor_to_multiple_of_8(width)
            height = self.floor_to_multiple_of_8(height)

            # 检查并调整以确保面积不超过目标面积
            while width * height > target_area:
                # 根据比例决定减少宽度还是高度
                if width / height > ratio:
                    width -= 8
                else:
                    height -= 8

                # 重新根据比例调整另一边
                if width / height > ratio:
                    height = self.floor_to_multiple_of_8(width / ratio)
                else:
                    width = self.floor_to_multiple_of_8(height * ratio)

        else:  # max_constrained 模式
            # 计算两种可能的尺寸
            width1 = expected_width
            height1 = width1 / ratio

            height2 = expected_height
            width2 = height2 * ratio

            # 选择不超过限制的最大尺寸
            if height1 <= expected_height:
                width = width1
                height = height1
            else:
                width = width2
                height = height2

            # 向下取整到8的倍数
            width = self.floor_to_multiple_of_8(width)
            height = self.floor_to_multiple_of_8(height)

            # 确保不超过预期尺寸
            while width > expected_width or height > expected_height:
                if width > expected_width:
                    width -= 8
                    height = self.floor_to_multiple_of_8(width / ratio)
                else:
                    height -= 8
                    width = self.floor_to_multiple_of_8(height * ratio)

        # 确保最小值
        width = max(64, width)
        height = max(64, height)

        return int(width), int(height)

    def calculate(self, mask, expected_width, expected_height, mode):
        """
        主计算函数
        """
        # 确保mask是NHWC格式
        mask = ensure_nhwc_mask_auto(mask)

        # 确保mask的通道数为1
        if mask.shape[3] != 1:
            raise ValueError("Mask must have 1 channel in NHWC format")

        # 提取mask的NHW部分并去除通道维度
        mask_nhw = mask.squeeze(3)

        # 获取mask的边界框
        boxes = masks_to_boxes(mask_nhw)

        # 计算宽高比
        min_x = boxes[:, 0]
        min_y = boxes[:, 1]
        max_x = boxes[:, 2]
        max_y = boxes[:, 3]

        # 计算mask的宽高和比例
        mask_width = (max_x - min_x + 1)[0].item()
        mask_height = (max_y - min_y + 1)[0].item()
        ratio = mask_width / mask_height

        # 计算新的宽高
        width, height = self.calculate_dimensions(
            ratio, expected_width, expected_height, mode
        )

        return (width, height)

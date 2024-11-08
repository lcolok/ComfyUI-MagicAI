import torch
import math
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torchvision.ops import masks_to_boxes
from magicai.utils.uniformers import ensure_nhwc_mask_auto
from magicai.utils.converters import (
    novel_mask2pil,
    novel_pil2tensor,
    novel_tensor2pil,
)


def tensor2rgba(image):
    """将图像张量转换为RGBA格式"""
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a tensor")

    if len(image.shape) != 4:
        raise ValueError("Input tensor must be 4D (B,H,W,C)")

    # 如果是RGB格式（3通道），添加alpha通道
    if image.shape[3] == 3:
        alpha = torch.ones((*image.shape[:-1], 1), device=image.device)
        return torch.cat((image, alpha), dim=3)
    elif image.shape[3] == 4:
        return image
    else:
        raise ValueError("Input tensor must have 3 or 4 channels")


class PasteByMask:
    """
    将 image_to_paste 基于 mask 贴到 image_base 上。
    resize_behavior 参数决定了如何调整待贴图像的大小以适应mask。
    如果使用来自'Separate Mask Components'节点的 mask_mapping_optional，
    它将控制哪个图像贴到哪个基础图像上。

    输入:
        image_base: NHWC 格式的基础图像
        image_to_paste: NHWC 格式的待贴图像
        mask: NHWC 格式的mask，其中C=1
        resize_behavior: 调整大小的行为
        mask_mapping_optional: 可选的mask映射
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("MASK",),  # 直接接受MASK类型
                "resize_behavior": (
                    [
                        "resize",
                        "keep_ratio_fill",
                        "keep_ratio_fit",
                        "source_size",
                        "source_size_unmasked",
                    ],
                ),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "LogicAI/Operation"

    def paste(
        self,
        image_base,
        image_to_paste,
        mask,
        resize_behavior,
        mask_mapping_optional=None,
    ):
        # 确保mask是NHWC格式
        mask = ensure_nhwc_mask_auto(mask)

        # 确保mask的通道数为1
        if mask.shape[3] != 1:
            raise ValueError("Mask must have 1 channel in NHWC format")

        # 转换为RGBA格式
        image_base = tensor2rgba(image_base)
        image_to_paste = tensor2rgba(image_to_paste)

        # 统一尺寸处理
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]

        if mask_mapping_optional is None:
            if B < PB:
                assert PB % B == 0
                image_base = image_base.repeat(PB // B, 1, 1, 1)
            B, H, W, C = image_base.shape
            if MB < B:
                assert B % MB == 0
                mask = mask.repeat(B // MB, 1, 1, 1)
            elif B < MB:
                assert MB % B == 0
                image_base = image_base.repeat(MB // B, 1, 1, 1)
            if PB < B:
                assert B % PB == 0
                image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)

        # 提取mask的NHW部分
        mask_nhw = mask.squeeze(3)

        # 调整mask大小
        mask_nhw = F.interpolate(mask_nhw.unsqueeze(1), size=(H, W), mode="nearest")[
            :, 0, :, :
        ]
        MB, MH, MW = mask_nhw.shape

        # 处理空mask的情况
        is_empty = ~torch.gt(
            torch.max(torch.reshape(mask_nhw, [MB, MH * MW]), dim=1).values, 0.0
        )
        mask_nhw[is_empty, 0, 0] = 1.0
        boxes = masks_to_boxes(mask_nhw)
        mask_nhw[is_empty, 0, 0] = 0.0

        # 计算边界框信息
        min_x = boxes[:, 0]
        min_y = boxes[:, 1]
        max_x = boxes[:, 2]
        max_y = boxes[:, 3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        for i in range(0, MB):
            if is_empty[i]:
                continue
            else:
                image_index = i
                if mask_mapping_optional is not None:
                    image_index = mask_mapping_optional[i].item()

                SB, SH, SW, _ = image_to_paste.shape

                # 计算目标尺寸
                width = int(target_width[i].item())
                height = int(target_height[i].item())

                # 根据不同的resize_behavior调整尺寸
                if resize_behavior == "keep_ratio_fill":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        width = int(height * actual_ratio)
                    elif actual_ratio < target_ratio:
                        height = int(width / actual_ratio)
                elif resize_behavior == "keep_ratio_fit":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        height = int(width / actual_ratio)
                    elif actual_ratio < target_ratio:
                        width = int(height * actual_ratio)
                elif (
                    resize_behavior == "source_size"
                    or resize_behavior == "source_size_unmasked"
                ):
                    width = SW
                    height = SH

                # 调整图像大小
                resized_image = image_to_paste[i].unsqueeze(0)
                if SH != height or SW != width:
                    resized_image = F.interpolate(
                        resized_image.permute(0, 3, 1, 2),
                        size=(height, width),
                        mode="bicubic",
                    ).permute(0, 2, 3, 1)

                # 准备贴图
                pasting = torch.zeros([H, W, C], device=image_base.device)
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                # 处理边界情况
                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0

                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= xmax - W
                    xmax = W
                if ymax > H:
                    source_ymax -= ymax - H
                    ymax = H

                # 执行贴图
                pasting[ymin:ymax, xmin:xmax, :] = resized_image[
                    0, source_ymin:source_ymax, source_xmin:source_xmax, :
                ]

                # 处理alpha通道和mask
                if resize_behavior == "keep_ratio_fill":
                    # 创建一个基于位置的mask
                    position_mask = torch.zeros([H, W], device=image_base.device)
                    position_mask[ymin:ymax, xmin:xmax] = 1.0
                    # 结合原始mask和位置mask
                    combined_mask = torch.min(position_mask, mask_nhw[i])
                    # 应用alpha通道
                    alpha_mask = pasting[:, :, 3]
                    final_mask = torch.min(combined_mask, alpha_mask)
                else:
                    # 其他模式保持原样
                    alpha_mask = torch.zeros([H, W], device=image_base.device)
                    alpha_mask[ymin:ymax, xmin:xmax] = resized_image[
                        0, source_ymin:source_ymax, source_xmin:source_xmax, 3
                    ]
                    if resize_behavior == "source_size_unmasked":
                        final_mask = alpha_mask
                    else:
                        final_mask = torch.min(alpha_mask, mask_nhw[i])

                # 扩展mask到4个通道
                final_mask = final_mask.unsqueeze(2).repeat(1, 1, 4)

                # 合并结果
                result[image_index] = pasting * final_mask + result[image_index] * (
                    1.0 - final_mask
                )

        return (result,)

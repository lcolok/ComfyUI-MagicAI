import random

import folder_paths  # type: ignore[import]
import torch
from magicai.utils.uniformers import ensure_nhwc_mask_auto
from nodes import SaveImage
from rich import print
from rich.style import Style

warning_style = Style(color="red", bold=True)


class MaskToPreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    RETURN_TYPES = ()
    FUNCTION = "mask_preview_image"

    OUTPUT_NODE = True

    CATEGORY = "LogicAI/IO"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def mask_to_image(self, masks):
        # 打印输入 masks 的形状，以便调试和理解数据
        # print("Masks shape:", masks.shape)

        # 确保 masks 是浮点型
        # 这是因为深度学习和图像处理通常需要高精度的浮点运算
        if not masks.dtype.is_floating_point:
            masks = masks.float()

        # 将 masks 转换为 NHWC 格式
        # print('传入的 masks 的形状为：', masks.shape)
        tensor = ensure_nhwc_mask_auto(masks)
        # print('转换为 NHWC 格式后的 masks 的形状为：', tensor.shape)
        if tensor is None:  # 如果返回的是 None，那么说明输入的形状是无效的
            return None

        # 如果是单通道图像，需要复制通道以转换为RGB格式
        # 通过判断 C 是否等于 1 来判定是否是单通道图像
        if tensor.shape[-1] == 1:
            tensor_rgb = torch.cat([tensor] * 3, dim=-1)
        else:
            tensor_rgb = tensor

        return tensor_rgb

    def mask_preview_image(
        self, masks, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        images = self.mask_to_image(masks)
        if images is None:
            print("[ERROR] Failed to convert masks to images. Returning empty list.")
            return [], []  # 返回空列表而不是 None

        # 调用父类的 save_images 方法
        return super().save_images(images, filename_prefix, prompt, extra_pnginfo)

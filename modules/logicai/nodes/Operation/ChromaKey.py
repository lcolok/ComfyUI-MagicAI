import torch
import numpy as np
from PIL import Image
import colorsys
import re
from logicai.utils.uniformers import ensure_nhwc_mask_auto
from logicai.utils.converters import (
    novel_pil2tensor,
    novel_tensor2pil,
)


class ChromaKeyToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "key_color": ("STRING", {"default": "#00FF00", "multiline": False}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "smoothing": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "hue_tolerance": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.3, "step": 0.01},
                ),
                "saturation_tolerance": (
                    "FLOAT",
                    {  # 改为饱和度容差
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "value_tolerance": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "white_clip": (
                    "FLOAT",
                    {  # 新增白色裁剪参数
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_chroma_key_mask"
    CATEGORY = "LogicAI/Operation"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if "key_color" not in kwargs:
            raise ValueError("ChromaKeyToMask - Error: key_color parameter is missing")

        hex_color = kwargs["key_color"].strip()
        pattern = r"^#?([A-Fa-f0-9]{6})$"

        if not re.match(pattern, hex_color):
            raise ValueError(
                f"ChromaKeyToMask - Error: Invalid hex color format '{hex_color}'. Expected format: #RRGGBB or RRGGBB"
            )

        return True

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def rgb_to_hsv_vectorized(self, rgb_image):
        rgb_normalized = rgb_image / 255.0

        r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)

        hsv_h = np.zeros_like(r)
        diff = maxc - minc

        mask = diff != 0

        mask_r = mask & (maxc == r)
        hsv_h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360

        mask_g = mask & (maxc == g)
        hsv_h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120

        mask_b = mask & (maxc == b)
        hsv_h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240

        hsv_h = hsv_h / 360.0

        hsv_s = np.zeros_like(r)
        mask = maxc != 0
        hsv_s[mask] = diff[mask] / maxc[mask]

        hsv_v = maxc

        return np.stack([hsv_h, hsv_s, hsv_v], axis=-1)

    def generate_chroma_key_mask(
        self,
        image,
        key_color,
        threshold,
        smoothing,
        hue_tolerance,
        saturation_tolerance,
        value_tolerance,
        white_clip,
        invert_mask,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError("ChromaKeyToMask - Error: Input image must be a tensor")

        image = ensure_nhwc_mask_auto(image)

        pattern = r"^#?([A-Fa-f0-9]{6})$"
        if not re.match(pattern, key_color.strip()):
            raise ValueError(
                f"ChromaKeyToMask - Error: Invalid hex color format '{key_color}'. Expected format: #RRGGBB or RRGGBB"
            )

        key_color = f"#{key_color.strip('#')}"

        try:
            target_rgb = self.hex_to_rgb(key_color)
            target_hsv = colorsys.rgb_to_hsv(*target_rgb)
        except Exception as e:
            raise ValueError(
                f"ChromaKeyToMask - Error: Failed to convert color '{key_color}' to HSV: {str(e)}"
            )

        ret_masks = []

        for img_tensor in image:
            try:
                img = novel_tensor2pil(torch.unsqueeze(img_tensor, 0))
                img_array = np.array(img.convert("RGB"))
                img_hsv = self.rgb_to_hsv_vectorized(img_array)

                # 计算色相差异（考虑循环特性）
                hue_diff = np.minimum(
                    np.abs(img_hsv[..., 0] - target_hsv[0]),
                    np.minimum(
                        np.abs(img_hsv[..., 0] - target_hsv[0] + 1),
                        np.abs(img_hsv[..., 0] - target_hsv[0] - 1),
                    ),
                )

                # 标准化色相差异
                normalized_hue_diff = hue_diff / hue_tolerance

                # 计算饱和度差异和权重
                sat_diff = np.abs(img_hsv[..., 1] - target_hsv[1])
                normalized_sat_diff = sat_diff / saturation_tolerance

                # 计算亮度差异
                val_diff = np.abs(img_hsv[..., 2] - target_hsv[2])
                normalized_val_diff = val_diff / value_tolerance

                # 创建高亮度/低饱和度的惩罚项（处理接近白色的区域）
                white_penalty = np.clip(
                    (img_hsv[..., 2] - white_clip) / (1 - white_clip), 0, 1
                ) * (1 - img_hsv[..., 1])

                # 综合计算距离，加入白色惩罚
                distance = (
                    np.minimum(normalized_hue_diff, 1.0) * 0.5  # 色相 (降低权重)
                    + np.minimum(normalized_sat_diff, 1.0) * 0.3  # 饱和度 (增加权重)
                    + np.minimum(normalized_val_diff, 1.0) * 0.2  # 亮度
                    + white_penalty * 0.5  # 白色惩罚项
                )

                # 创建alpha遮罩
                alpha = np.zeros_like(distance)

                # 应用阈值和平滑处理
                alpha[distance < threshold] = 0
                smooth_mask = (distance >= threshold) & (
                    distance < threshold + smoothing
                )
                alpha[smooth_mask] = (
                    (distance[smooth_mask] - threshold) / smoothing * 255
                ).astype(np.uint8)
                alpha[distance >= threshold + smoothing] = 255

                if invert_mask:
                    alpha = 255 - alpha

                mask_img = Image.fromarray(alpha.astype(np.uint8), mode="L")
                mask_tensor = novel_pil2tensor(mask_img)
                mask_tensor = ensure_nhwc_mask_auto(mask_tensor)
                ret_masks.append(mask_tensor)

            except Exception as e:
                raise ValueError(
                    f"ChromaKeyToMask - Error: Failed to process image: {str(e)}"
                )

        try:
            final_mask = torch.cat(ret_masks, dim=0)
            final_mask = ensure_nhwc_mask_auto(final_mask)

            return (final_mask,)

        except Exception as e:
            raise ValueError(
                f"ChromaKeyToMask - Error: Failed to combine masks: {str(e)}"
            )

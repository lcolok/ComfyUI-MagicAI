from magicai.utils.converters import (
    novel_tensor2pil,
    novel_pil2tensor,
)

from magicai.configs import MAX_RESOLUTION
from magicai.nodes.Composite.utils.upscale_base import (
    upscale,
    calculate_resized_dimensions,
    get_image_dimensions,
    resize_to_expected_size,
    upscale,
    RESIZE_STRATEGY_DEFAULT,
    RESIZE_STRATEGY_OPTIONS,
)

# 策略常量选项和描述
RESIZE_STRATEGY_DESCRIPTION = """支持两种策略：
    - Maintain Pixels: 保持总像素接近期望值，保持宽高比
    - Within Bounds: 在保持宽高比的同时，确保宽高都不超过期望值"""


class ResizeToExpectedSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_to_resize": ("IMAGE",),
                "exp_width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "exp_height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_to_expected_size_exec"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = f"调整图像大小到指定尺寸。{RESIZE_STRATEGY_DESCRIPTION}"

    def resize_to_expected_size_exec(
        self, image_to_resize, exp_width, exp_height, strategy
    ):
        # Round expected dimensions to multiple of 8
        target_width, target_height = get_image_dimensions(exp_width, exp_height)
        resized_img = resize_to_expected_size(
            image_to_resize, target_width, target_height, strategy
        )
        return (resized_img.unsqueeze(0),)


class ResizeToMatchSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "image_to_resize": ("IMAGE",),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_to_match_size"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = f"调整图像大小以匹配参考图像的尺寸。{RESIZE_STRATEGY_DESCRIPTION}"

    def resize_to_match_size(self, reference_image, image_to_resize, strategy):
        # Convert images to PIL
        reference_img = novel_tensor2pil(reference_image)
        image_to_resize_pil = novel_tensor2pil(image_to_resize)

        # Get reference dimensions and round to multiple of 8
        ref_width, ref_height = reference_img.size
        target_width, target_height = get_image_dimensions(ref_width, ref_height)

        # Resize image to match the size of the reference image with specified strategy
        resized_img = resize_to_expected_size(
            image_to_resize, target_width, target_height, strategy
        )

        return (resized_img.unsqueeze(0),)


class ResizeToExpectedSizeWithUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "exp_width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "exp_height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "upscale_mode": (
                    ["AUTO", "Force Upscale", "Never Upscale"],
                    {"default": "AUTO"},
                ),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = ("upscaled_image", "resized_image")
    FUNCTION = "upscale_and_resize"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = (
        f"使用上采样模型放大图像后调整到指定尺寸。{RESIZE_STRATEGY_DESCRIPTION}"
    )

    def upscale_and_resize(
        self, upscale_model, image, exp_width, exp_height, upscale_mode, strategy
    ):
        _, img_width, img_height, _ = image.shape

        # Round expected dimensions to multiple of 8
        target_width, target_height = get_image_dimensions(exp_width, exp_height)

        # 检查 image 是否为浮点型
        if not image.dtype.is_floating_point:
            image = image.float()

        # 检查图像是否需要放大
        need_upscale = img_width * img_height < target_width * target_height

        # 根据模式和需要选择是否放大图像
        if upscale_mode == "Force Upscale" or (upscale_mode == "AUTO" and need_upscale):
            try:
                upscaled_img = upscale(upscale_model, image)
            except Exception as e:
                raise e
        else:
            upscaled_img = image

        # 调整图像大小到期望尺寸，使用指定的策略
        try:
            resized_img = resize_to_expected_size(
                upscaled_img, target_width, target_height, strategy
            )
        except Exception as e:
            raise e

        return (upscaled_img, resized_img.unsqueeze(0))


class ResizeToMatchSizeWithUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "reference_image": ("IMAGE",),
                "image_to_resize": ("IMAGE",),
                "upscale_mode": (
                    ["AUTO", "Force Upscale", "Never Upscale"],
                    {"default": "AUTO"},
                ),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = ("upscaled_image", "resized_image")
    FUNCTION = "upscale_and_resize_match_size"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = f"使用上采样模型放大图像后调整到匹配参考图像的尺寸。{RESIZE_STRATEGY_DESCRIPTION}"

    def upscale_and_resize_match_size(
        self, upscale_model, reference_image, image_to_resize, upscale_mode, strategy
    ):
        _, img_width, img_height, _ = image_to_resize.shape
        _, ref_width, ref_height, _ = reference_image.shape

        # Round reference dimensions to multiple of 8
        target_width, target_height = get_image_dimensions(ref_width, ref_height)

        # 检查 image 是否为浮点型
        if not image_to_resize.dtype.is_floating_point:
            image_to_resize = image_to_resize.float()

        upscaled_img = image_to_resize  # set a default value

        # Force Upscale
        if upscale_mode == "Force Upscale":
            try:
                upscaled_img = upscale(upscale_model, image_to_resize)
            except Exception as e:
                raise e
        # AUTO: upscale only when necessary
        elif (
            upscale_mode == "AUTO"
            and img_width * img_height < target_width * target_height
        ):
            try:
                upscaled_img = upscale(upscale_model, image_to_resize)
            except Exception as e:
                raise e

        # 使用指定的策略调整大小
        try:
            resized_img = resize_to_expected_size(
                upscaled_img, target_width, target_height, strategy
            )
        except Exception as e:
            raise e

        return (upscaled_img, resized_img.unsqueeze(0))


def resize_mask_to_expected_size(
    mask_to_resize, exp_width, exp_height, strategy=RESIZE_STRATEGY_DEFAULT
):
    """
    调整掩码大小的专用函数，保持掩码的归一化值
    """
    # 转换掩码为PIL图像进行调整大小，保持值范围
    mask_np = mask_to_resize.cpu().numpy()
    mask_pil = novel_tensor2pil(mask_to_resize)

    # 获取原始尺寸
    img_width, img_height = mask_pil.size

    # 计算新尺寸，使用指定的策略
    new_width, new_height = calculate_resized_dimensions(
        img_width, img_height, exp_width, exp_height, strategy
    )

    # 使用双线性插值调整大小以保持平滑过渡
    resized_mask = mask_pil.resize((new_width, new_height), resample=2)  # BILINEAR

    # 转换回tensor，保持值范围
    return novel_pil2tensor(resized_mask)


class MaskResizeToExpectedSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_to_resize": ("MASK",),
                "exp_width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "exp_height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize_mask_to_expected_size_exec"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = (
        f"调整掩码大小到指定尺寸，保持掩码的归一化值范围。{RESIZE_STRATEGY_DESCRIPTION}"
    )

    def resize_mask_to_expected_size_exec(
        self, mask_to_resize, exp_width, exp_height, strategy
    ):
        # 确保期望尺寸是8的倍数
        target_width, target_height = get_image_dimensions(exp_width, exp_height)

        # 调整掩码大小，保持归一化值
        resized_mask = resize_mask_to_expected_size(
            mask_to_resize, target_width, target_height, strategy
        )

        return (resized_mask.unsqueeze(0),)


class MaskResizeToMatchSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "mask_to_resize": ("MASK",),
                "strategy": (
                    RESIZE_STRATEGY_OPTIONS,
                    {"default": RESIZE_STRATEGY_DEFAULT},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize_mask_to_match_size"
    CATEGORY = "MagicAI/Composite"
    DESCRIPTION = f"调整掩码大小以匹配参考图像的尺寸，保持掩码的归一化值范围。{RESIZE_STRATEGY_DESCRIPTION}"

    def resize_mask_to_match_size(self, reference_image, mask_to_resize, strategy):
        # 转换参考图像为PIL以获取尺寸
        reference_img = novel_tensor2pil(reference_image)

        # 获取参考尺寸并确保是8的倍数
        ref_width, ref_height = reference_img.size
        target_width, target_height = get_image_dimensions(ref_width, ref_height)

        # 调整掩码大小，保持归一化值，使用指定的策略
        resized_mask = resize_mask_to_expected_size(
            mask_to_resize, target_width, target_height, strategy
        )

        return (resized_mask.unsqueeze(0),)

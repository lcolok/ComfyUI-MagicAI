import comfy
import torch
from magicai.utils.converters import (
    novel_tensor2pil,
    novel_pil2tensor,
)

RESIZE_STRATEGY_OPTIONS = ["Maintain Pixels", "Within Bounds"]
RESIZE_STRATEGY_DEFAULT = "Maintain Pixels"


def upscale(upscale_model, image):
    """使用给定的模型对图像进行上采样"""
    device = comfy.model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 128 + 64
    overlap = 8
    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )
    pbar = comfy.utils.ProgressBar(steps)
    s = comfy.utils.tiled_scale(
        in_img,
        lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=pbar,
    )
    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    return s


def resize_to_expected_size(
    image_to_resize, exp_width, exp_height, strategy=RESIZE_STRATEGY_DEFAULT
):
    """
    调整图像到期望大小，使用指定的调整策略
    """
    # Convert images to PIL
    image_to_resize = novel_tensor2pil(image_to_resize)

    # Get original image dimensions
    img_width, img_height = image_to_resize.size

    # Calculate the new dimensions using the helper function
    new_width, new_height = calculate_resized_dimensions(
        img_width, img_height, exp_width, exp_height, strategy
    )

    # Resize image to the new dimensions
    resized_img = image_to_resize.resize((new_width, new_height))

    return novel_pil2tensor(resized_img)


def calculate_resized_dimensions(
    img_width,
    img_height,
    exp_width,
    exp_height,
    strategy=RESIZE_STRATEGY_DEFAULT,
    tolerance=5,
):
    """
    根据给定的原始图像尺寸、期望尺寸和策略，计算新的宽和高

    策略:
    - Maintain Pixels: 保持总像素接近期望值，保持宽高比
    - Within Bounds: 在保持宽高比的同时，确保宽高都不超过期望值
    """
    aspect_ratio = img_width / img_height

    if strategy == "Maintain Pixels":
        # 原有的策略：保持总像素数接近期望值
        total_pixels = exp_width * exp_height
        new_width = round((total_pixels * aspect_ratio) ** 0.5)
        new_height = round(total_pixels / new_width)

        # 检查是否接近期望尺寸
        if abs(new_width - exp_width) <= tolerance:
            new_width = exp_width
        if abs(new_height - exp_height) <= tolerance:
            new_height = exp_height

    elif strategy == "Within Bounds":
        # 新策略：确保宽高都不超过期望值
        width_ratio = exp_width / img_width
        height_ratio = exp_height / img_height

        # 使用较小的缩放比例以确保两个维度都不超过期望值
        scale = min(width_ratio, height_ratio)

        # 计算新的尺寸
        new_width = round(img_width * scale)
        new_height = round(img_height * scale)

    # 确保是8的倍数
    return get_image_dimensions(new_width, new_height)


def round_to_multiple_of_8(dimension):
    """将尺寸向下取整到最接近的8的倍数"""
    return dimension - (dimension % 8)


def get_image_dimensions(width, height):
    """获取符合8的倍数的图像尺寸"""
    return round_to_multiple_of_8(width), round_to_multiple_of_8(height)

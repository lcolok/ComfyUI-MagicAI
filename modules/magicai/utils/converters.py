import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
import comfy
import tempfile
import io
import base64

DEFAULT_DEVICE = torch.device("cpu")
TORCH_DEVICE = comfy.model_management.get_torch_device()


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def novel_tensor2pil(image):
    image = image.clamp(0, 1)  # 将张量限制在0到1之间
    image = (255.0 * image).byte()  # 乘以255并转换为字节类型
    image = image.cpu().numpy().squeeze()  # 转换为NumPy数组并去除多余的维度
    return Image.fromarray(image)


# Tensor to Numpy
def tensor2numpy(image):
    return np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)


def novel_tensor2numpy(image):
    return image.clamp(0, 1).mul(255).byte().cpu().numpy().squeeze()


def numpy2tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255.0)


def novel_numpy2tensor(image):
    return torch.from_numpy(image).float().div(255).to(DEFAULT_DEVICE)


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)


def novel_pil2tensor(image):
    tensor = torch.from_numpy(np.array(image))
    tensor = tensor.float().div(255).to(DEFAULT_DEVICE)
    return tensor


def combine_individual_masks(masks):
    if len(masks) == 0:
        raise ValueError("The 'masks' list cannot be empty.")

    combined = torch.zeros_like(masks[0])
    for mask in masks:
        if mask.shape != combined.shape:
            raise ValueError("All masks must have the same shape.")
        combined += mask

    combined = torch.clamp(combined, 0, 1)

    return combined


# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype("uint8")
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def novel_mask2pil(mask):
    """
    将掩码张量转换为PIL图像，修正了squeeze的应用。
    参数:
        mask (Tensor): 要转换的掩码张量。其形状应为[C, H, W]或[H, W]。
    返回:
        PIL.Image: 转换后的PIL图像。
    """
    # 确保掩码在0和1之间
    mask = torch.clamp(mask, 0, 1)
    # 转换为字节类型
    mask = mask.mul(255).byte()
    # 转换为NumPy数组
    mask_np = mask.cpu().numpy()
    # 检查形状是否为[H, W]或[C, H, W]，如果是前者，则添加一个批次维度，以便后续处理
    if mask_np.shape[0] == 1:
        # 处理后的形状为[1, H, W]
        mask_np = mask_np.squeeze(0)
    # 使用"L"模式（灰度）创建PIL图像
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def tensor_image_to_base64(img):
    img_pil = novel_tensor2pil(img)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def transform_image_to_tensor(image_data):
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[0])))
    image_tensor = novel_pil2tensor(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def save_tensor_image_to_temp_file(tensor_image, format="PNG"):
    """
    将给定的图像张量保存为临时文件。

    :param tensor_image: 要保存的图像张量。
    :param format: 保存图像的格式，默认为 PNG。
    :return: 临时文件的路径。
    """
    # 将张量转换为 PIL 图像
    image = novel_tensor2pil(tensor_image)

    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix="." + format.lower(), delete=False)
    # 保存图像到临时文件
    image.save(temp_file, format=format)
    # 获取文件名
    temp_file_path = temp_file.name
    # 关闭文件以确保写入完成
    temp_file.close()

    return temp_file_path


def save_batch_tensor_images_to_temp_files(batch_tensor_image, format="PNG"):
    """
    将给定的批量图像张量中的每张图像保存为临时文件。

    :param batch_tensor_image: 要保存的批量图像张量。
    :param format: 保存图像的格式，默认为 PNG。
    :return: 一个包含所有临时文件路径的列表。
    """
    temp_files_paths = []
    for i in range(batch_tensor_image.shape[0]):
        # 获取单张图像的张量
        single_tensor_image = batch_tensor_image[i]
        # 使用已有函数保存单张图像
        temp_file_path = save_tensor_image_to_temp_file(single_tensor_image, format)
        # 将临时文件路径添加到列表中
        temp_files_paths.append(temp_file_path)

    return temp_files_paths


def optimized_mask_to_uint8(mask):
    """
    根据输入遮罩的所在设备，高效地将其转换为uint8类型。

    参数:
    mask: 可能是NumPy数组或PyTorch张量，任意形状。

    返回:
    mask_uint8: 转换为uint8类型并适当压缩维度的NumPy数组。
    """
    try:
        # 如果输入是PyTorch张量
        if isinstance(mask, torch.Tensor):
            # 在GPU上进行计算
            if mask.device.type == "cuda":
                # 将值限制在0到1之间
                mask_clamped = torch.clamp(mask, 0, 1)
                # 转换为uint8类型
                mask_uint8 = (mask_clamped * 255).to(torch.uint8)
            # 如果已在CPU上
            else:
                mask_clamped = torch.clamp(mask, 0, 1)
                mask_uint8 = (mask_clamped * 255).to(torch.uint8)

            # 将张量从GPU移至CPU并转换为NumPy数组
            mask_uint8 = mask_uint8.cpu().numpy()
        # 如果输入是NumPy数组
        elif isinstance(mask, np.ndarray):
            mask_clamped = np.clip(mask, 0, 1)
            if mask_clamped.dtype.kind == "f":
                mask_clamped = mask_clamped.astype(np.float32)
            mask_uint8 = (mask_clamped * 255).astype(np.uint8)
        else:
            raise ValueError("输入的mask需要是numpy.ndarray或torch.Tensor!")

        # 检查维度压缩是否需要且有效
        if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
            mask_uint8 = mask_uint8.squeeze(0)

        return mask_uint8

    except Exception as e:
        print(f"在优化遮罩至uint8过程中发生错误: {repr(e)}")
        return None


def numpy_to_tensor(array, dtype=torch.float32, device=None):
    """
    将NumPy数组转换为PyTorch张量。

    参数:
    array (numpy.ndarray): 要转换的NumPy数组。
    dtype (torch.dtype, 可选): 结果张量的数据类型，默认为torch.float32。
    device (str, 可选): 目标设备，默认为'cpu'。

    返回:
    torch.Tensor: 转换后的PyTorch张量。
    """
    device = device or DEFAULT_DEVICE
    tensor = torch.from_numpy(array).to(dtype=dtype, device=device)
    return tensor

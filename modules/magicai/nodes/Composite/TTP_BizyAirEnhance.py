from magicai.configs import MAX_RESOLUTION
from magicai.nodes.Composite.utils.upscale_base import (
    resize_to_expected_size, RESIZE_STRATEGY_DEFAULT, RESIZE_STRATEGY_OPTIONS
)
from magicai.utils.uniformers import ensure_nhwc_mask_auto
from bizyengine.core import BizyAirNodeIO
from bizyengine.core.data_types import UPSCALE_MODEL
from bizyengine.bizyair_extras.nodes_upscale_model import ImageUpscaleWithModel
import numpy as np
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 辅助函数 - 基础工具
def ensure_divisible_by_8(value):
    """确保值是 8 的倍数"""
    return (value // 8) * 8

def calculate_optimal_factors(width, height, min_factor):
    """根据图像宽高比计算最优的宽高切分因子"""
    aspect_ratio = width / height
    width_factor = min_factor
    height_factor = min_factor

    if aspect_ratio > 1:  # 宽图
        if 1.4 <= aspect_ratio <= 1.6:
            width_factor, height_factor = max(3, min_factor), max(2, min_factor)
        elif aspect_ratio >= 1.3:
            if aspect_ratio < 1.8:
                width_factor, height_factor = max(3, min_factor), max(2, min_factor)
            else:
                width_factor, height_factor = max(4, min_factor), max(2, min_factor)
    elif aspect_ratio < 1:  # 高图
        if aspect_ratio <= 0.75:
            if aspect_ratio > 0.55:
                width_factor, height_factor = max(2, min_factor), max(3, min_factor)
            else:
                width_factor, height_factor = max(2, min_factor), max(4, min_factor)

    # 限制过度切分
    max_allowed = min_factor + 2
    width_factor = min(width_factor, max_allowed)
    height_factor = min(height_factor, max_allowed)

    return width_factor, height_factor

def calculate_square_tile_factors(width, height):
    """
    计算正方形分块的最优方案
    目标: 用最少数量的正方形tile完全覆盖图像

    返回: (width_factor, height_factor, tile_size)
    """
    min_tile_size = 384  # BizyAir API 最小要求

    # 从最大可能的正方形尺寸开始
    # 理想的正方形边长应该基于较小的维度
    shorter_side = min(width, height)

    # 尝试不同的正方形尺寸，找到最优方案
    best_tile_size = None
    min_total_tiles = float('inf')
    best_cols = 0
    best_rows = 0

    # 从较大的tile尺寸开始尝试（减少分块数）
    # 最大tile尺寸不超过 shorter_side，最小不低于 384
    max_tile_size = ensure_divisible_by_8(shorter_side)

    for tile_size in range(max_tile_size, min_tile_size - 1, -8):  # 每次减少8
        if tile_size < min_tile_size:
            break

        # 计算需要的分块数
        cols = (width + tile_size - 1) // tile_size
        rows = (height + tile_size - 1) // tile_size
        total_tiles = cols * rows

        # 找到最少分块数的方案
        if total_tiles < min_total_tiles:
            min_total_tiles = total_tiles
            best_tile_size = tile_size
            best_cols = cols
            best_rows = rows

        # 如果已经找到很少的分块数，可以提前退出
        if total_tiles <= 4:
            break

    # 如果没有找到合适的方案，使用最小尺寸
    if best_tile_size is None:
        best_tile_size = min_tile_size
        best_cols = (width + min_tile_size - 1) // min_tile_size
        best_rows = (height + min_tile_size - 1) // min_tile_size

    return best_cols, best_rows, best_tile_size

def get_model_name(upscale_model):
    """从upscale_model对象中获取模型名称"""
    if isinstance(upscale_model, BizyAirNodeIO):
        try:
            if hasattr(upscale_model, "model_name"):
                return upscale_model.model_name
            elif hasattr(upscale_model, "name"):
                return upscale_model.name
            else:
                model_name = str(upscale_model)
                return "RealESRGAN_x4plus" if "<" in model_name and ">" in model_name else model_name
        except:
            return "RealESRGAN_x4plus"
    else:
        return upscale_model

def ensure_image_tensor(image, fallback_image=None):
    """确保图像是PyTorch张量"""
    if image is None:
        if fallback_image is not None:
            return ensure_image_tensor(fallback_image)
        else:
            raise ValueError("图像为None且没有提供备用图像")
    
    if isinstance(image, torch.Tensor):
        image = image.float()
        if image.device.type != 'cpu':
            image = image.cpu()
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    else:
        try:
            image_np = np.array(image, dtype=np.float32)
            image = torch.from_numpy(image_np).float()
        except Exception as e1:
            try:
                image = torch.tensor(image, dtype=torch.float32)
            except Exception as e2:
                if fallback_image is not None:
                    return ensure_image_tensor(fallback_image)
                else:
                    raise TypeError(f"无法将图像转换为PyTorch张量: {type(image)}")
    
    if len(image.shape) > 4:
        while len(image.shape) > 4:
            image = image.squeeze(0)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    return image

# 图像分割与合并函数
def split_image_for_bizyair(image, max_size=1280, width_factor=None, height_factor=None):
    """将大图像分割成小块"""
    print(f"[split_image_for_bizyair] 开始分割图像...")
    image = ensure_nhwc_mask_auto(image)
    _, height, width, channels = image.shape
    print(f"[split_image_for_bizyair] 图像尺寸: {width}x{height}, channels={channels}")

    if width <= max_size and height <= max_size:
        print(f"[split_image_for_bizyair] 图像小于 {max_size}，无需分割")
        return [image], [(0, 0, width, height)]

    # BizyAir API 最小tile尺寸要求
    min_tile_size = 384

    if width_factor is not None and height_factor is not None:
        cols, rows = width_factor, height_factor
        print(f"[split_image_for_bizyair] 初始分割因子: 宽度分块数 {cols} x 高度分块数 {rows}")

        # 计算初始tile尺寸
        tile_width = (width + cols - 1) // cols
        tile_height = (height + rows - 1) // rows
        tile_width = ensure_divisible_by_8(tile_width)
        tile_height = ensure_divisible_by_8(tile_height)

        # 检查并调整过小的tile尺寸
        if tile_width < min_tile_size or tile_height < min_tile_size:
            print(f"[split_image_for_bizyair] ⚠️  计算的tile尺寸 {tile_width}x{tile_height} 小于最小要求 {min_tile_size}")

            # 减少分块数量直到满足最小尺寸要求
            while (tile_width < min_tile_size or tile_height < min_tile_size) and (cols > 1 or rows > 1):
                if tile_width < min_tile_size and cols > 1:
                    cols -= 1
                if tile_height < min_tile_size and rows > 1:
                    rows -= 1

                tile_width = (width + cols - 1) // cols
                tile_height = (height + rows - 1) // rows
                tile_width = ensure_divisible_by_8(tile_width)
                tile_height = ensure_divisible_by_8(tile_height)

            print(f"[split_image_for_bizyair] 调整后分割因子: 宽度分块数 {cols} x 高度分块数 {rows}")
    else:
        cols = (width + max_size - 1) // max_size
        rows = (height + max_size - 1) // max_size
        print(f"[split_image_for_bizyair] 图像尺寸超过 {max_size}x{max_size}，将分割为 {rows}x{cols} 块")
        tile_width = ensure_divisible_by_8(max_size)
        tile_height = ensure_divisible_by_8(max_size)

    print(f"[split_image_for_bizyair] 最终分块尺寸: {tile_width}x{tile_height} (分块数: {cols}x{rows}={cols*rows}个)")
    tiles = []
    tile_positions = []
    
    try:
        for r in range(rows):
            for c in range(cols):
                start_y = r * tile_height
                start_x = c * tile_width
                end_y = min(start_y + tile_height, height)
                end_x = min(start_x + tile_width, width)
                
                if end_y <= start_y or end_x <= start_x:
                    continue
                
                tile = image[:, start_y:end_y, start_x:end_x, :]
                tiles.append(tile)
                tile_positions.append((start_x, start_y, end_x - start_x, end_y - start_y))
    except Exception as e:
        print(f"[split_image_for_bizyair] 分割图像时出错: {e}")
        if len(tiles) == 0:
            tiles = [image]
            tile_positions = [(0, 0, width, height)]

    print(f"[split_image_for_bizyair] 分割完成，共 {len(tiles)} 个图像块")
    return tiles, tile_positions

def merge_upscaled_tiles(tiles, tile_positions, original_shape, scale_factor=4):
    """合并上采样后的图像块"""
    try:
        if not tiles or len(tiles) != len(tile_positions):
            raise ValueError("图像块数量与位置信息不匹配")
        
        _, orig_height, orig_width, channels = original_shape
        upscaled_height = orig_height * scale_factor
        upscaled_width = orig_width * scale_factor
        
        result = torch.zeros((1, upscaled_height, upscaled_width, channels), dtype=tiles[0].dtype)
        
        for i, (tile, (x, y, w, h)) in enumerate(zip(tiles, tile_positions)):
            upscaled_x = x * scale_factor
            upscaled_y = y * scale_factor
            upscaled_w = tile.shape[2]
            upscaled_h = tile.shape[1]
            
            if upscaled_h == 0 or upscaled_w == 0:
                continue
                
            if upscaled_y + upscaled_h > upscaled_height:
                upscaled_h = upscaled_height - upscaled_y
            if upscaled_x + upscaled_w > upscaled_width:
                upscaled_w = upscaled_width - upscaled_x
            
            result[:, upscaled_y:upscaled_y+upscaled_h, upscaled_x:upscaled_x+upscaled_w, :] = tile[:, :upscaled_h, :upscaled_w, :]
        
        return result
    except Exception as e:
        print(f"合并图像块时出错: {e}")
        return tiles[0] if tiles and len(tiles) > 0 else torch.zeros((1, orig_height, orig_width, channels), dtype=torch.float32)

def calculate_dimensions(exp_width, exp_height, width_factor, height_factor, overlap_rate):
    """计算目标图像尺寸和分块尺寸"""
    exp_width = ensure_divisible_by_8(exp_width)
    exp_height = ensure_divisible_by_8(exp_height)
    
    overlap_width_pixels = int(exp_width * overlap_rate)
    overlap_height_pixels = int(exp_height * overlap_rate)
    
    target_width = exp_width + (width_factor - 1) * (exp_width - overlap_width_pixels)
    target_height = exp_height + (height_factor - 1) * (exp_height - overlap_height_pixels)
    
    target_width = ensure_divisible_by_8(target_width + 7)
    target_height = ensure_divisible_by_8(target_height + 7)
    
    return target_width, target_height, overlap_width_pixels, overlap_height_pixels

def calculate_tile_dimensions(res_width, res_height, width_factor, height_factor, overlap_width_pixels, overlap_height_pixels):
    """计算单个分块的尺寸"""
    if width_factor == 1:
        tile_width = res_width
    else:
        tile_width = int((res_width + (width_factor - 1) * overlap_width_pixels) / width_factor)
    
    if height_factor == 1:
        tile_height = res_height
    else:
        tile_height = int((res_height + (height_factor - 1) * overlap_height_pixels) / height_factor)
    
    return ensure_divisible_by_8(tile_width), ensure_divisible_by_8(tile_height)

# 基类，包含共享处理逻辑
class BaseBizyAirUpscaler:
    """BizyAir图像放大处理的基类"""

    def _process_single_tile_with_retry(self, tile, tile_index, total_tiles, upscale_model, model_name, scale_factor=4):
        """处理单个图像块，包含改进的重试机制和降级方案"""
        max_retries = 5  # 增加重试次数
        base_retry_delay = 2.0  # 基础延迟增加到2秒

        tile_shape = tile.shape
        print(f"[并发处理] 正在处理第 {tile_index+1}/{total_tiles} 个图像块 (尺寸: {tile_shape[2]}x{tile_shape[1]})...")

        for retry in range(max_retries):
            try:
                # 每次重试都重新创建 upscaler 对象，避免状态污染
                upscaler = ImageUpscaleWithModel()
                setattr(upscaler, "_assigned_id", f"tile_{tile_index}_{retry}")

                if retry > 0:
                    # 递增延迟：2秒, 3秒, 4秒, 5秒, 6秒
                    delay = base_retry_delay + retry
                    print(f"[并发处理] 第 {tile_index+1}/{total_tiles} 块 - 第 {retry+1}/{max_retries} 次重试 (等待{delay}秒)...")
                    time.sleep(delay)

                start_time = time.time()

                if isinstance(upscale_model, BizyAirNodeIO):
                    result = upscaler.default_function(upscale_model=upscale_model, image=tile)
                else:
                    result = upscaler.default_function(upscale_model=model_name, image=tile)

                elapsed = time.time() - start_time

                upscaled_tile = result[0] if isinstance(result, tuple) and len(result) > 0 else result
                upscaled_tile = ensure_image_tensor(upscaled_tile)
                upscaled_tile = ensure_nhwc_mask_auto(upscaled_tile)

                print(f"[并发处理] 第 {tile_index+1}/{total_tiles} 块处理完成 (耗时: {elapsed:.2f}秒)")
                return upscaled_tile

            except Exception as e:
                print(f"[并发处理] 第 {tile_index+1}/{total_tiles} 块第 {retry+1} 次尝试失败: {e}")
                if retry == max_retries - 1:
                    # 所有重试失败，使用降级方案：torch 插值放大
                    print(f"[并发处理] ⚠️  第 {tile_index+1}/{total_tiles} 块所有 {max_retries} 次重试失败")
                    print(f"[并发处理] 使用降级方案: torch 插值放大 {scale_factor}x 以保持尺寸一致")
                    traceback.print_exc()

                    # 使用双三次插值放大到目标尺寸
                    try:
                        import torch.nn.functional as F
                        # tile shape: (1, H, W, C) -> (1, C, H, W) for interpolate
                        tile_nchw = tile.permute(0, 3, 1, 2)
                        target_h = tile_shape[1] * scale_factor
                        target_w = tile_shape[2] * scale_factor
                        upscaled_nchw = F.interpolate(
                            tile_nchw,
                            size=(target_h, target_w),
                            mode='bicubic',
                            align_corners=False
                        )
                        # 转回 NHWC 并裁剪到 [0, 1] 范围（bicubic可能产生超出范围的值）
                        upscaled_tile = upscaled_nchw.permute(0, 2, 3, 1)
                        upscaled_tile = torch.clamp(upscaled_tile, 0.0, 1.0)
                        print(f"[并发处理] 降级放大完成: {tile_shape[2]}x{tile_shape[1]} → {target_w}x{target_h}")
                        return upscaled_tile
                    except Exception as fallback_error:
                        print(f"[并发处理] ❌ 降级方案也失败: {fallback_error}")
                        # 最后的保底：返回原始tile的4倍放大版本（使用最简单的方法）
                        return tile.repeat(1, scale_factor, scale_factor, 1)

        # 理论上不会到这里，但保险起见
        return tile

    def upscale_with_bizyair(self, upscale_model, image, width_factor=None, height_factor=None, batch_size=3):
        """使用BizyAir进行图像上采样"""
        print(f"[upscale_with_bizyair] 开始上采样 (width_factor={width_factor}, height_factor={height_factor})")
        try:
            model_name = get_model_name(upscale_model)
            print(f"[upscale_with_bizyair] 模型名称: {model_name}")
            image = ensure_image_tensor(image)
            image = ensure_nhwc_mask_auto(image)

            _, height, width, _ = image.shape
            print(f"[upscale_with_bizyair] 输入图像尺寸: {width}x{height}")
            max_size = 1280

            # 大图像分割处理
            if width > max_size or height > max_size:
                print(f"[upscale_with_bizyair] 图像超过 {max_size}，使用分块并发处理")
                tiles, tile_positions = split_image_for_bizyair(
                    image, max_size, width_factor, height_factor
                )

                total_tiles = len(tiles)
                print(f"开始并发处理 {total_tiles} 个图像块 (每批 {batch_size} 个)...")

                batch_start_time = time.time()

                # 使用 ThreadPoolExecutor 进行并发处理
                upscaled_tiles = [None] * total_tiles  # 预分配列表保持顺序

                # 确定缩放因子（通常是4x）
                scale_factor = 4

                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # 提交所有任务
                    future_to_index = {
                        executor.submit(
                            self._process_single_tile_with_retry,
                            tile, i, total_tiles, upscale_model, model_name, scale_factor
                        ): i
                        for i, tile in enumerate(tiles)
                    }

                    # 收集结果（保持顺序）
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            upscaled_tiles[index] = future.result()
                        except Exception as e:
                            print(f"[并发处理] 第 {index+1} 块处理异常: {e}")
                            print(f"[并发处理] 使用 torch 插值作为最终降级方案")
                            # 使用 torch 插值放大保持尺寸一致
                            import torch.nn.functional as F
                            tile = tiles[index]
                            tile_nchw = tile.permute(0, 3, 1, 2)
                            target_h = tile.shape[1] * scale_factor
                            target_w = tile.shape[2] * scale_factor
                            upscaled_nchw = F.interpolate(tile_nchw, size=(target_h, target_w), mode='bicubic', align_corners=False)
                            # 裁剪到 [0, 1] 范围避免黑图
                            upscaled_tile = upscaled_nchw.permute(0, 2, 3, 1)
                            upscaled_tiles[index] = torch.clamp(upscaled_tile, 0.0, 1.0)

                batch_elapsed = time.time() - batch_start_time
                print(f"所有图像块并发处理完成 (总耗时: {batch_elapsed:.2f}秒, 平均: {batch_elapsed/total_tiles:.2f}秒/块)...")
                
                if not upscaled_tiles:
                    return image
                
                # 计算缩放因子并合并分块
                scale_factor = 4
                if upscaled_tiles[0].shape[1] > tile_positions[0][3]:
                    scale_factor = upscaled_tiles[0].shape[1] // tile_positions[0][3]

                print(f"所有图像块处理完成，开始合并 {len(upscaled_tiles)} 个图像块，缩放因子={scale_factor}...")
                merged = merge_upscaled_tiles(upscaled_tiles, tile_positions, image.shape, scale_factor)
                print(f"图像块合并完成")
                return merged
            
            # 小图像直接处理
            else:
                print(f"[upscale_with_bizyair] 图像小于 {max_size}，直接处理")
                scale_factor = 4  # 默认4x放大
                upscaled_img = self._process_single_tile_with_retry(image, 0, 1, upscale_model, model_name, scale_factor)
                print(f"[upscale_with_bizyair] 上采样完成，返回结果")
                return upscaled_img

        except Exception as e:
            print(f"[upscale_with_bizyair] 上采样过程出错: {e}")
            import traceback
            traceback.print_exc()
            return image

    def process_image_common(self, image, exp_width, exp_height, width_factor, height_factor,
                            overlap_rate, upscale_model, upscale_mode, strategy):
        """通用图像处理逻辑"""
        print(f"\n[process_image_common] 开始通用处理流程")
        print(f"[process_image_common] 分块因子: width_factor={width_factor}, height_factor={height_factor}")

        image = ensure_nhwc_mask_auto(image)
        _, img_height, img_width, _ = image.shape
        print(f"[process_image_common] 输入图像尺寸: {img_width}x{img_height}")

        # 计算目标尺寸
        print(f"[process_image_common] 计算目标尺寸...")
        target_width, target_height, overlap_width_pixels, overlap_height_pixels = calculate_dimensions(
            exp_width, exp_height, width_factor, height_factor, overlap_rate
        )
        print(f"[process_image_common] 目标尺寸: {target_width}x{target_height}")
        
        # 决定是否需要放大
        need_upscale = img_width * img_height < target_width * target_height
        is_large_image = img_width > 1280 or img_height > 1280
        upscaled_img = image
        
        # 进行放大处理
        if upscale_mode == "Force Upscale" or (upscale_mode == "AUTO" and need_upscale):
            try:
                print(f"开始 BizyAir 上采样处理 (is_large_image={is_large_image}, width_factor={width_factor}, height_factor={height_factor})...")
                if is_large_image:
                    upscaled_img = self.upscale_with_bizyair(upscale_model, image, width_factor, height_factor, batch_size=3)
                else:
                    upscaled_img = self.upscale_with_bizyair(upscale_model, image, batch_size=3)
                print(f"BizyAir 上采样完成")
            except Exception as e:
                print(f"BizyAir 上采样失败: {e}")
                upscaled_img = image
        
        # 调整到目标尺寸
        try:
            print(f"开始调整图像大小到目标尺寸 ({target_width}x{target_height})...")
            upscaled_img = ensure_image_tensor(upscaled_img, fallback_image=image)
            resized_img = resize_to_expected_size(upscaled_img, target_width, target_height, strategy)
            resized_img = ensure_nhwc_mask_auto(resized_img)
            print(f"图像大小调整完成")
        except Exception as e:
            print(f"图像大小调整失败: {e}")
            resized_img = image
        
        # 计算最终分块尺寸
        _, res_height, res_width, _ = resized_img.shape
        print(f"[process_image_common] 最终调整后图像尺寸: {res_width}x{res_height}")

        tile_width, tile_height = calculate_tile_dimensions(
            res_width, res_height, width_factor, height_factor,
            overlap_width_pixels, overlap_height_pixels
        )
        print(f"[process_image_common] 分块尺寸: {tile_width}x{tile_height}")

        # 计算分块宽高比
        tile_aspect_ratio = tile_width / tile_height
        print(f"[process_image_common] 分块宽高比: {tile_aspect_ratio:.2f}")

        print(f"[process_image_common] 处理完成，准备返回结果\n")
        return (upscaled_img, resized_img, target_width, target_height, tile_width,
                tile_height, width_factor, height_factor, tile_aspect_ratio)

# 自动计算分块因子的节点
class Tile_ExpectedImageSize_MagicAI(BaseBizyAirUpscaler):
    """使用BizyAir云端资源的自动分块因子图像放大节点 (MagicAI)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exp_width": ("INT", {"default": 1280, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                "exp_height": ("INT", {"default": 1280, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                "min_split": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "overlap_rate": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 0.95, "step": 0.05}),
                "upscale_model": (UPSCALE_MODEL,),
                "upscale_mode": (["AUTO", "Force Upscale", "Never Upscale"], {"default": "AUTO"}),
                "strategy": (RESIZE_STRATEGY_OPTIONS, {"default": RESIZE_STRATEGY_DEFAULT}),
                "force_square_tiles": (["false", "true"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("upscaled_image", "resized_image", "target_width", "target_height", 
                    "tile_width", "tile_height", "width_factor", "height_factor", "tile_aspect_ratio")
    CATEGORY = "MagicAI/BizyAir_Hijack/TTP"
    FUNCTION = "process_image"
    DESCRIPTION = "使用BizyAir云端资源根据期望的分块大小进行图像放大处理，自动计算最优分块因子 (MagicAI)"

    def process_image(self, image, exp_width, exp_height, min_split, overlap_rate,
                     upscale_model, upscale_mode, strategy, force_square_tiles):
        """处理图像，自动计算最优分块数量"""
        print(f"\n{'='*60}")
        print(f"[Tile_ExpectedImageSize_MagicAI] 开始处理图像")
        print(f"参数: exp_width={exp_width}, exp_height={exp_height}, min_split={min_split}")
        print(f"      overlap_rate={overlap_rate}, upscale_mode={upscale_mode}")
        print(f"      strategy={strategy}, force_square_tiles={force_square_tiles}")
        print(f"{'='*60}\n")

        image = ensure_nhwc_mask_auto(image)
        _, img_height, img_width, _ = image.shape
        print(f"[process_image] 输入图像尺寸: {img_width}x{img_height}")

        # 计算最佳分块因子
        if force_square_tiles == "true":
            print(f"[process_image] 使用正方形分块模式 (忽略 min_split 参数)")
            # 使用新的正方形分块算法
            width_factor, height_factor, optimal_tile_size = calculate_square_tile_factors(img_width, img_height)
            actual_tile_width = (img_width + width_factor - 1) // width_factor
            actual_tile_height = (img_height + height_factor - 1) // height_factor
            print(f"[process_image] 正方形分块方案: {width_factor}x{height_factor}={width_factor*height_factor}块")
            print(f"[process_image] 最优tile尺寸: {optimal_tile_size}x{optimal_tile_size} (实际: {actual_tile_width}x{actual_tile_height})")
        else:
            print(f"[process_image] 计算最佳分块因子 (基于宽高比)...")
            width_factor, height_factor = calculate_optimal_factors(img_width, img_height, min_split)
            tile_width_estimate = img_width / width_factor
            tile_height_estimate = img_height / height_factor
            print(f"[process_image] 初始分块因子: {width_factor}x{height_factor}={width_factor*height_factor}块")
            print(f"[process_image] 预估tile尺寸: {tile_width_estimate:.0f}x{tile_height_estimate:.0f}")

        # 调用共享处理逻辑
        print(f"[process_image] 调用共享处理逻辑...")
        result = self.process_image_common(
            image, exp_width, exp_height, width_factor, height_factor,
            overlap_rate, upscale_model, upscale_mode, strategy
        )
        print(f"[process_image] 节点处理完成！\n{'='*60}\n")
        return result

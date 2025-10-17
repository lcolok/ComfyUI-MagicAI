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

    def _process_single_tile_with_retry(self, tile, tile_index, total_tiles, upscale_model, model_name):
        """处理单个图像块，包含重试机制"""
        max_retries = 3
        retry_delay = 1.0

        tile_shape = tile.shape
        print(f"[并发处理] 正在处理第 {tile_index+1}/{total_tiles} 个图像块 (尺寸: {tile_shape[2]}x{tile_shape[1]})...")

        upscaler = ImageUpscaleWithModel()
        setattr(upscaler, "_assigned_id", "12345")

        for retry in range(max_retries):
            try:
                if retry > 0:
                    print(f"[并发处理] 第 {tile_index+1}/{total_tiles} 块 - 第 {retry+1}/{max_retries} 次重试...")
                    time.sleep(retry_delay)

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
                    print(f"[并发处理] 第 {tile_index+1}/{total_tiles} 块所有重试失败，使用原始图块")
                    traceback.print_exc()
                    return tile

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

                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # 提交所有任务
                    future_to_index = {
                        executor.submit(
                            self._process_single_tile_with_retry,
                            tile, i, total_tiles, upscale_model, model_name
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
                            upscaled_tiles[index] = tiles[index]  # 使用原始tile

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
                upscaled_img = self._process_single_tile_with_retry(image, 0, 1, upscale_model, model_name)
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
        print(f"[process_image] 计算最佳分块因子...")
        width_factor, height_factor = calculate_optimal_factors(img_width, img_height, min_split)
        print(f"[process_image] 初始分块因子: width_factor={width_factor}, height_factor={height_factor}")
        
        # 处理正方形分块选项
        if force_square_tiles == "true":
            # 迭代调整直到宽高比接近正方形，但限制最大迭代次数防止卡死
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                tile_aspect = (img_width / width_factor) / (img_height / height_factor)

                # 如果已经足够接近正方形，退出
                if 0.8 <= tile_aspect <= 1.2:
                    break

                # 调整因子
                if tile_aspect > 1.2:  # 分块太宽
                    if width_factor < 8:
                        width_factor += 1
                    elif height_factor > min_split:
                        height_factor -= 1
                    else:
                        # 无法继续调整，退出循环
                        break
                elif tile_aspect < 0.8:  # 分块太高
                    if height_factor < 8:
                        height_factor += 1
                    elif width_factor > min_split:
                        width_factor -= 1
                    else:
                        # 无法继续调整，退出循环
                        break

                iteration += 1

            # 打印调试信息
            final_aspect = (img_width / width_factor) / (img_height / height_factor)
            print(f"[process_image] force_square_tiles: 迭代 {iteration} 次, width_factor={width_factor}, height_factor={height_factor}, tile_aspect={final_aspect:.2f}")

        # 调用共享处理逻辑
        print(f"[process_image] 调用共享处理逻辑...")
        result = self.process_image_common(
            image, exp_width, exp_height, width_factor, height_factor,
            overlap_rate, upscale_model, upscale_mode, strategy
        )
        print(f"[process_image] 节点处理完成！\n{'='*60}\n")
        return result

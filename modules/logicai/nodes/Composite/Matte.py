import torch
from PIL import Image, ImageOps
from logicai.utils.converters import (
    novel_tensor2pil,
    novel_pil2tensor,
)


class AlphaMatte:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": (
                    "BOOLEAN",
                    {"default": False, "label_on": "True", "label_off": "False"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_blend_mask"

    CATEGORY = "LogicAI/Composite"

    def image_blend_mask(self, image, mask, invert_mask):
        # 初始化结果列表
        results = []

        # 迭代处理每一对图像和遮罩
        for elem_image, elem_mask in zip(image, mask):
            # 将图像和遮罩转换为PIL格式
            img = novel_tensor2pil(elem_image)
            elem_mask = novel_tensor2pil(elem_mask.squeeze(0)).convert("L")

            # 如果需要，反转遮罩
            if invert_mask:
                elem_mask = ImageOps.invert(elem_mask)

            # 使用遮罩合成图像
            masked_img = Image.composite(
                img, Image.new("RGBA", img.size), elem_mask.resize(img.size)
            )

            # 将合成后的图像转换为RGBA格式，并转换回张量格式
            img_result = novel_pil2tensor(masked_img.convert("RGBA"))

            # 将结果添加到结果列表中
            results.append(img_result)

        # 使用torch.stack来堆叠所有结果张量，新增一个维度
        return (torch.stack(results, dim=0),)

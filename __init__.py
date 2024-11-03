"""
@author: LogicAI
@title: ComfyUI-MagicAI
@nickname: 🧠 LogicAI
"""

import os
import sys

# from .setup import setup_all
# setup_all() # 每次启动Comfy都会执行setup_all()，这样非常耗时，所以注释掉

current_path = os.path.abspath(os.path.dirname(__file__))
modules_path = os.path.join(current_path, "modules")

if os.path.isdir(modules_path):
    sys.path.insert(0, modules_path)

from .modules.logicai.nodes.Composite import *
from .modules.logicai.nodes.IO import *
from .modules.logicai.nodes.Operation import *
from .modules.logicai.utils.mappings import *


# Category Composite
composite_mappings = [
    (AlphaMatte, "🎨 AlphaMatte (MagicAI)", "🎨 Alpha Matte (MagicAI)"),
]

# Category IO
io_mappings = [
    (TextBox, "✍🏻 TextBox (MagicAI)", "✍🏻 Text Box (MagicAI)"),
    (MaskToPreviewImage, "👹 PreviewMask (MagicAI)", "👹 Preview Mask (MagicAI)"),
]

# Category Operation
operation_mappings = [
    (
        MaskSizeCalculator,
        "👹🖩 Mask Size Calculator (MagicAI)",
        "👹🖩 Mask Size Calculator (MagicAI)",
    ),
    (
        UniversalMaskConverter,
        "🔀 UniversalMaskConverter (MagicAI)",
        "🔀 Universal Mask Converter (MagicAI)",
    ),
    (ChromaKeyToMask, "🎨👹 ChromaKeyToMask (MagicAI)", "🎨👹 Chroma Key To Mask (MagicAI)"),
]

# Define mappings as a list of category mappings
mappings = [
    composite_mappings,
    io_mappings,
    operation_mappings,
]


# Combine all mappings into a single list
all_mappings = []
for category in mappings:
    all_mappings.extend(category)

# Generate mappings and assign to variables
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_mappings(all_mappings)

# web ui的节点功能
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

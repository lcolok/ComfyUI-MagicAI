import json

import ftfy
from jinja2 import BaseLoader, Environment
from server import PromptServer


class JinjaTemplateRenderer:
    """
    Jinja2模板渲染器节点
    用于将JSON数据应用到Jinja2模板中，并输出处理后的文本
    包含文本清理和格式化功能
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数类型

        Returns:
            dict: 包含必需的模板文本和JSON字符串输入字段
        """
        return {
            "required": {
                # 多行文本输入，用于Jinja2模板
                "template": ("STRING", {"multiline": True}),
                # 多行文本输入，用于JSON数据
                "json_string": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    # 定义输出类型为字符串
    RETURN_TYPES = ("STRING",)
    # 定义输出名称
    RETURN_NAMES = ("processed_prompt",)
    # 定义处理函数名
    FUNCTION = "process_template"
    # 定义节点类别
    CATEGORY = "LogicAI/Feature Fusion"

    @staticmethod
    def clean_output(text):
        """
        使用ftfy清理和规范化处理后的模板输出

        Args:
            text (str): 原始模板输出文本

        Returns:
            str: 清理后的文本
        """
        # 使用ftfy修复文本编码和格式问题
        cleaned_text = ftfy.fix_text(text)

        # 规范化空白字符，将多个空格合并为单个空格
        cleaned_text = " ".join(cleaned_text.split())

        # 规范化逗号后的空格
        cleaned_text = cleaned_text.replace(" , ", ", ")

        # 去除首尾空白字符
        return cleaned_text.strip()

    def process_template(self, template, json_string):
        """
        处理Jinja2模板和JSON数据，生成最终文本

        Args:
            template (str): Jinja2模板字符串
            json_string (str): JSON格式的数据字符串

        Returns:
            tuple: 包含处理后文本的元组

        Raises:
            ValueError: 当JSON数据格式无效时
            Exception: 模板处理过程中的其他错误
        """
        try:
            # 解析JSON数据
            data = json.loads(json_string)
        except json.JSONDecodeError:
            raise ValueError("JSON数据格式无效")

        # 创建Jinja2环境，配置优化设置
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,  # 移除块级标签后的第一个换行
            lstrip_blocks=True,  # 移除块级标签前的空白
            keep_trailing_newline=False,  # 不保留末尾换行
        )

        def join_filter(value, d=", "):
            """
            自定义过滤器：将列表项连接为字符串

            Args:
                value: 要处理的值（预期为列表）
                d (str): 连接符，默认为", "

            Returns:
                str: 连接后的字符串
            """
            if isinstance(value, list):
                # 过滤None值和空字符串，转换为字符串
                valid_items = [
                    str(item)
                    for item in value
                    if item is not None and str(item).strip()
                ]
                return d.join(valid_items)
            return str(value) if value is not None else ""

        # 注册自定义过滤器
        env.filters["join"] = join_filter
        env.filters["fix_text"] = ftfy.fix_text  # 添加ftfy文本修复过滤器

        try:
            # 渲染模板
            template = env.from_string(template)
            processed_prompt = template.render(data)

            # 清理输出文本
            processed_prompt = self.clean_output(processed_prompt)

            return (processed_prompt,)

        except Exception as e:
            print(f"模板处理错误: {str(e)}")
            raise

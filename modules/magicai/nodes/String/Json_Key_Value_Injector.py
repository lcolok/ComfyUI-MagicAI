import json
from server import PromptServer
from magicai.utils.types import AlwaysEqualProxy


any_type = AlwaysEqualProxy("*")


class JsonKeyValueInjector:
    """
    JSON键值对注入器节点
    用于向JSON字符串中添加新的键值对，并返回更新后的JSON字符串
    json_string为强制输入，value支持任意类型
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数类型

        Returns:
            dict: 包含必需的JSON字符串、键名和键值输入字段
        """
        return {
            "required": {
                # 原始JSON字符串输入，设置为强制输入
                "json_string": ("STRING", {"multiline": True, "forceInput": True}),
                # 要添加的键名
                "key": ("STRING", {"multiline": False}),
                # 要添加的键值，支持任意类型输入
                "value": (any_type, {}),
            },
        }

    # 定义输出类型为字符串
    RETURN_TYPES = ("STRING",)
    # 定义输出名称
    RETURN_NAMES = ("updated_json_string",)
    # 定义处理函数名
    FUNCTION = "inject_key_value"
    # 定义节点类别
    CATEGORY = "LogicAI/Feature Fusion"

    def validate_json(self, json_string):
        """
        验证JSON字符串的有效性

        Args:
            json_string (str): 待验证的JSON字符串

        Returns:
            dict: 解析后的JSON数据

        Raises:
            ValueError: 当JSON格式无效时抛出异常
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式无效: {str(e)}")

    def inject_key_value(self, json_string, key, value):
        """
        向JSON数据中注入新的键值对

        Args:
            json_string (str): 原始JSON字符串
            key (str): 要添加的键名
            value (any): 要添加的键值，支持任意类型

        Returns:
            tuple: 包含更新后JSON字符串的元组
        """
        try:
            # 验证并解析原始JSON
            data = self.validate_json(json_string)

            # 直接将value添加到数据中，不需要额外的类型判断
            # 因为json.dumps会自动处理Python的基本数据类型
            data[key] = value

            # 将更新后的数据转换回JSON字符串，保持格式美观
            updated_json_string = json.dumps(
                data,
                ensure_ascii=False,  # 允许非ASCII字符
                indent=2,  # 使用2空格缩进
                separators=(",", ": "),  # 设置分隔符格式
                default=str,  # 对于无法序列化的对象，转换为字符串
            )

            return (updated_json_string,)

        except Exception as e:
            print(f"键值对注入错误: {str(e)}")
            raise

import json

from magicai.utils.types import AlwaysEqualProxy
from server import PromptServer


class AlwaysEqualProxy(str):
    """
    特殊的字符串代理类，用于比较操作
    无论与什么值比较都返回True（相等）或False（不等）
    """

    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class JsonKeyValueInjector:
    """
    JSON键值对注入器节点
    用于向JSON字符串中添加新的键值对，支持通过点号表示的嵌套路径
    自动识别和转换输入值的类型，包括JSON字符串的自动解析
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数类型
        """
        return {
            "required": {
                # 原始JSON字符串输入，设置为强制输入
                "json_string": ("STRING", {"multiline": True, "forceInput": True}),
                # 要添加的键路径，支持点号分隔的嵌套路径
                "key": ("STRING", {"multiline": False}),
                # 要添加的键值，支持任意类型输入
                "value": (AlwaysEqualProxy("*"), {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_json_string",)
    FUNCTION = "inject_key_value"
    CATEGORY = "LogicAI/Feature Fusion"

    def validate_json(self, json_string):
        """
        验证JSON字符串的有效性
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式无效: {str(e)}")

    def smart_type_conversion(self, value):
        """
        智能类型转换
        尝试将输入值转换为最合适的类型

        Args:
            value: 输入值

        Returns:
            转换后的值
        """
        # 如果输入已经是字典或列表类型，直接返回
        if isinstance(value, (dict, list)):
            return value

        # 如果是字符串类型，尝试进行JSON解析
        if isinstance(value, str):
            # 去除首尾空白
            value = value.strip()

            # 尝试解析JSON
            try:
                if (
                    value.startswith("{")
                    and value.endswith("}")
                    or value.startswith("[")
                    and value.endswith("]")
                ):
                    return json.loads(value)
            except json.JSONDecodeError:
                pass

            # 尝试转换为数字
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                pass

            # 处理布尔值
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            if value.lower() == "null":
                return None

        # 其他情况返回原值
        return value

    def set_nested_value(self, data, key_path, value):
        """
        在嵌套的字典结构中设置值

        Args:
            data (dict): 要修改的字典
            key_path (str): 点号分隔的键路径 (例如: "a.b.c")
            value (any): 要设置的值
        """
        keys = key_path.split(".")
        current = data

        # 遍历除最后一个键以外的所有键
        for key in keys[:-1]:
            # 如果键不存在或对应的值不是字典，则创建新的字典
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # 在设置值之前进行智能类型转换
        converted_value = self.smart_type_conversion(value)
        current[keys[-1]] = converted_value

        return data

    def inject_key_value(self, json_string, key, value):
        """
        向JSON数据中注入新的键值对，支持嵌套路径
        会自动识别和转换value的类型

        Args:
            json_string (str): 原始JSON字符串
            key (str): 键路径，支持点号分隔（例如: "user.address.city"）
            value (any): 要添加的键值，支持任意类型
        """
        try:
            # 验证并解析原始JSON
            data = self.validate_json(json_string)

            # 处理嵌套键值对，包含智能类型转换
            data = self.set_nested_value(data, key, value)

            # 转换回JSON字符串
            updated_json_string = json.dumps(
                data, ensure_ascii=False, indent=2, separators=(",", ": "), default=str
            )

            return (updated_json_string,)

        except Exception as e:
            print(f"键值对注入错误: {str(e)}")
            raise

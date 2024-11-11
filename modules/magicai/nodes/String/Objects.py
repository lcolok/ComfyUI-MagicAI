import json
from json_repair import repair_json


class ExtractJSONFromTextNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "format_output": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract_json"
    CATEGORY = "LogicAI/String"

    def extract_json(self, text, format_output):
        try:
            # 直接使用 json_repair 修复整个输入文本
            repaired_json = repair_json(text)

            # 解析修复后的 JSON
            json_obj = json.loads(repaired_json)

            # 根据 format_output 参数决定是否格式化输出
            if format_output:
                result = json.dumps(json_obj, indent=4, ensure_ascii=False)
            else:
                result = json.dumps(json_obj, ensure_ascii=False)

            return (result,)
        except Exception as e:
            # 如果修复和解析都失败,返回空对象
            print(f"Error extracting JSON: {str(e)}")
            return ("{}",)


import json
from json_repair import repair_json


class GetValueFromJsonString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True, "multiline": True}),
                "key": ("STRING", {"default": "mana"}),
                "fallback_value": ("STRING", {"default": "0"}),
                "is_enable": ("BOOLEAN", {"default": True}),
                "format_nested": (
                    "BOOLEAN",
                    {"default": True},
                ),  # 新增：控制嵌套JSON的格式化
            }
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "value_int",
        "value_str",
    )
    FUNCTION = "get_value"
    CATEGORY = "LogicAI/String"

    def _repair_and_parse_json(self, json_string):
        """尝试修复并解析JSON字符串"""
        try:
            # 首先尝试直接解析
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                # 如果直接解析失败，使用json_repair修复
                repaired_json = repair_json(json_string)
                return json.loads(repaired_json)
            except Exception as e:
                print(f"JSON修复和解析失败: {str(e)}")
                return None

    def _get_nested_value(self, data, key_path):
        """获取嵌套的值"""
        current = data
        keys = key_path.split(".")

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                index = int(key)
                current = current[index] if 0 <= index < len(current) else None
            else:
                return None

            if current is None:
                return None

        return current

    def _process_value(self, value, format_nested=True):
        """处理不同类型的值，返回(int_value, str_value)"""
        if value is None:
            return 0, "0"

        if isinstance(value, (dict, list)):
            if format_nested:
                str_value = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                str_value = json.dumps(value, ensure_ascii=False)
            return 0, str_value

        if isinstance(value, bool):
            return (1 if value else 0), str(value).lower()

        if isinstance(value, (int, float)):
            int_value = int(value)
            return int_value, str(value)

        # 字符串处理
        str_value = str(value)
        try:
            int_value = int(float(str_value))
            return int_value, str_value
        except (ValueError, TypeError):
            return 0, str_value

    def get_value(
        self, json_string, key, fallback_value="0", is_enable=True, format_nested=True
    ):
        """
        从JSON字符串中提取并处理指定键的值

        Args:
            json_string: JSON格式的字符串
            key: 要获取的键路径（支持点号分隔的嵌套路径）
            fallback_value: 获取失败时的默认值
            is_enable: 是否启用该节点
            format_nested: 是否格式化嵌套的JSON输出

        Returns:
            tuple: (整数值, 字符串值)
        """
        if not is_enable:
            return 0, "0"

        # 处理空输入
        if not json_string or not json_string.strip():
            return 0, fallback_value

        # 修复并解析JSON
        data = self._repair_and_parse_json(json_string)
        if data is None:
            return 0, fallback_value

        # 获取嵌套值
        value = self._get_nested_value(data, key)
        if value is None:
            return 0, fallback_value

        # 处理并返回值
        return self._process_value(value, format_nested)

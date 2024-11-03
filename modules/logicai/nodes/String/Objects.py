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


class GetValueFromJsonString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True, "multiline": True}),
                "key": ("STRING", {"default": "mana"}),
                "fallback_value": ("STRING", {"default": "0"}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
    )  # 返回两种类型以适应不同场景
    RETURN_NAMES = (
        "value_int",
        "value_str",
    )
    FUNCTION = "get_value"
    CATEGORY = "LogicAI/String"

    def get_value(self, json_string, key, fallback_value="0", is_enable=True):
        """
        从JSON字符串中提取指定键的值

        参数:
        json_string: JSON格式的字符串
        key: 要获取的键名
        fallback_value: 当获取失败时的默认值
        is_enable: 是否启用该节点

        返回:
        tuple: (整数值, 字符串值)
        """
        if not is_enable:
            return (
                0,
                "0",
            )

        try:
            # 尝试解析JSON
            data = json.loads(json_string)

            # 支持嵌套键访问（使用点号分隔）
            keys = key.split(".")
            value = data
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, fallback_value)
                elif isinstance(value, list) and k.isdigit():
                    value = (
                        value[int(k)] if 0 <= int(k) < len(value) else fallback_value
                    )
                else:
                    value = fallback_value
                    break

            # 处理不同类型的值
            if isinstance(value, (dict, list)):
                # 如果值是字典或列表，转换为格式化的JSON字符串
                str_value = json.dumps(value, ensure_ascii=False, indent=2)
                return (
                    0,
                    str_value,
                )
            elif isinstance(value, bool):
                return (
                    1 if value else 0,
                    str(value).lower(),
                )
            elif isinstance(value, (int, float)):
                return (
                    int(value),
                    str(value),
                )
            else:
                # 其他情况转换为字符串
                str_value = str(value)
                # 尝试转换为整数
                try:
                    int_value = int(float(str_value))
                    return (
                        int_value,
                        str_value,
                    )
                except (ValueError, TypeError):
                    return (
                        0,
                        str_value,
                    )

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            return (
                0,
                fallback_value,
            )
        except Exception as e:
            print(f"处理错误: {str(e)}")
            return (
                0,
                fallback_value,
            )

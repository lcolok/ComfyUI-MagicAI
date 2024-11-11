import textwrap
import ast
import os
import sys
import json
from json_repair import repair_json


class ExecutionContext:
    """用于管理执行环境的上下文对象"""

    def __init__(self, seed=0, json_data=None):
        self._variables = {"seed": seed}
        # 如果提供了JSON数据，将其添加到变量中
        if json_data is not None:
            if isinstance(json_data, dict):
                self._variables.update(json_data)
            else:
                self._variables["json_data"] = json_data

    def set(self, name, value):
        """设置变量"""
        self._variables[name] = value

    def get(self, name, default=None):
        """获取变量，支持默认值"""
        return self._variables.get(name, default)

    def get_all(self):
        """获取所有变量"""
        return self._variables


class PythonExecutionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "python_code": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "json_string": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute_python"
    CATEGORY = "utils"

    def _repair_and_parse_json(self, json_string):
        """尝试修复并解析JSON字符串"""
        if not json_string or not json_string.strip():
            return None

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

    def _check_code_safety(self, code):
        """检查代码是否包含禁止的导入"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 检查 import 语句
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name == "BizyAir.auth" or name.name.startswith(
                            "BizyAir.auth."
                        ):
                            return (
                                False,
                                "Direct import of BizyAir.auth module is not allowed",
                            )
                # 检查 from import 语句
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "BizyAir.auth"
                        or node.module.startswith("BizyAir.auth.")
                    ):
                        return False, "Import from BizyAir.auth module is not allowed"
        except Exception as e:
            return False, f"Code analysis error: {str(e)}"
        return True, ""

    def execute_python(self, python_code, seed, json_string=None):
        # 使用textwrap.dedent()来修复缩进
        dedented_code = textwrap.dedent(python_code)

        # 首先进行安全检查
        is_safe, message = self._check_code_safety(dedented_code)
        if not is_safe:
            return (f"Error: {message}",)

        # 解析JSON数据（如果提供）
        json_data = (
            self._repair_and_parse_json(json_string)
            if json_string is not None
            else None
        )

        # 创建执行上下文对象，并传入seed和JSON数据
        ctx = ExecutionContext(seed=seed, json_data=json_data)

        # 创建用于执行的命名空间
        local_vars = {}

        try:
            # 解析代码以检查语法错误
            ast.parse(dedented_code)

            # 设置全局命名空间
            global_vars = {
                "os": os,
                "sys": sys,
                "__file__": __file__,
                "__name__": "__main__",
                "ctx": ctx,  # 注入ctx对象
            }

            # 执行代码
            exec(dedented_code, global_vars, local_vars)

            # 获取最后一个表达式的结果
            tree = ast.parse(dedented_code)
            last_expr = next(
                (node for node in reversed(tree.body) if isinstance(node, ast.Expr)),
                None,
            )

            if last_expr:
                result = eval(
                    compile(ast.Expression(last_expr.value), "<string>", "eval"),
                    global_vars,
                    local_vars,
                )
            else:
                result = (
                    "Code executed successfully, but no expression result to return."
                )

        except Exception as e:
            result = f"Error: {str(e)}"

        return (str(result),)

import textwrap
import ast
import os
import sys
import json
from json_repair import repair_json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionContext:
    """用于管理执行环境的上下文对象"""

    seed: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后自动执行，确保seed存在于variables中"""
        self.variables["seed"] = self.seed

    def get(self, name: str, default: Any = None) -> Any:
        """获取变量值，支持默认值"""
        return self.variables.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """设置变量值"""
        self.variables[name] = value


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

    def _repair_and_parse_json(
        self, json_string: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """尝试修复并解析JSON字符串"""
        if not json_string or not json_string.strip():
            return None

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                repaired_json = repair_json(json_string)
                return json.loads(repaired_json)
            except Exception as e:
                print(f"JSON修复和解析失败: {str(e)}")
                return None

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """检查代码是否包含禁止的导入"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name == "BizyAir.auth" or name.name.startswith(
                            "BizyAir.auth."
                        ):
                            return (
                                False,
                                "Direct import of BizyAir.auth module is not allowed",
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "BizyAir.auth"
                        or node.module.startswith("BizyAir.auth.")
                    ):
                        return False, "Import from BizyAir.auth module is not allowed"
        except Exception as e:
            return False, f"Code analysis error: {str(e)}"
        return True, ""

    def execute_python(
        self, python_code: str, seed: int, json_string: Optional[str] = None
    ) -> tuple[str]:
        # 修复缩进
        dedented_code = textwrap.dedent(python_code)

        # 安全检查
        is_safe, message = self._check_code_safety(dedented_code)
        if not is_safe:
            return (f"Error: {message}",)

        # 创建上下文
        ctx = ExecutionContext(seed=seed)

        # 处理JSON数据
        if json_string:
            json_data = self._repair_and_parse_json(json_string)
            if isinstance(json_data, dict):
                ctx.variables.update(json_data)
            elif json_data is not None:
                ctx.set("json_data", json_data)

        try:
            # 解析代码检查语法
            ast.parse(dedented_code)

            # 执行环境
            global_vars = {
                "os": os,
                "sys": sys,
                "__file__": __file__,
                "__name__": "__main__",
                "ctx": ctx,
                "json": json,
            }
            local_vars = {}

            # 执行代码
            exec(dedented_code, global_vars, local_vars)

            # 获取最后的表达式结果
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

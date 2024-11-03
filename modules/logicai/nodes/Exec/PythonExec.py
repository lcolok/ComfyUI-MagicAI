import textwrap
import ast


class PythonExecutionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "python_code": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute_python"
    CATEGORY = "utils"

    def execute_python(self, python_code, seed):
        # 使用textwrap.dedent()来修复缩进
        dedented_code = textwrap.dedent(python_code)

        # 创建一个新的局部命名空间来执行代码
        local_vars = {"SEED": seed}

        try:
            # 解析代码以检查语法错误
            ast.parse(dedented_code)

            # 执行代码
            exec(dedented_code, {}, local_vars)

            # 获取最后一个表达式的结果
            tree = ast.parse(dedented_code)
            last_expr = next(
                (node for node in reversed(tree.body) if isinstance(node, ast.Expr)),
                None,
            )

            if last_expr:
                result = eval(
                    compile(ast.Expression(last_expr.value), "<string>", "eval"),
                    {},
                    local_vars,
                )
            else:
                result = (
                    "Code executed successfully, but no expression result to return."
                )

        except Exception as e:
            result = f"Error: {str(e)}"

        return (str(result),)

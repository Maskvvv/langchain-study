"""
第十一章：Agent 安全与 Guardrails — 工具安全边界

本脚本不依赖真实模型，重点演示工具层应该如何做安全控制。

运行方式：
    uv run python src/ch11_guardrails_security/01_tool_guardrails.py
"""

import ast
import operator
from dataclasses import dataclass


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression: str) -> float:
    """只允许四则运算的安全计算器。

    不使用 eval，避免执行任意 Python 代码。
    """

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return SAFE_OPERATORS[type(node.op)](left, right)

        raise ValueError("只允许数字和 + - * / 四则运算")

    tree = ast.parse(expression, mode="eval")
    return float(eval_node(tree))


@dataclass
class ToolRequest:
    """模拟模型生成的工具调用请求。"""

    tool_name: str
    args: dict
    risk_level: str


def require_approval(request: ToolRequest) -> bool:
    """判断工具调用是否需要人工审批。"""
    return request.risk_level == "high"


def execute_tool(request: ToolRequest, approved: bool = False) -> str:
    """执行工具前统一做安全检查。"""
    if require_approval(request) and not approved:
        return f"已拦截高风险工具：{request.tool_name}，等待人工审批"

    if request.tool_name == "calculate":
        result = safe_calculate(request.args["expression"])
        return f"计算结果：{result}"

    if request.tool_name == "send_email":
        to = request.args.get("to", "")
        if not to.endswith("@company.com"):
            return "拒绝发送：只允许发送到 company.com 邮箱"
        return f"邮件已发送给 {to}"

    return "未知工具，拒绝执行"


def main() -> None:
    print("=" * 60)
    print("第十一章：工具安全边界")
    print("=" * 60)

    safe_request = ToolRequest(
        tool_name="calculate",
        args={"expression": "(100 + 20) / 3"},
        risk_level="low",
    )
    print(execute_tool(safe_request))

    dangerous_request = ToolRequest(
        tool_name="send_email",
        args={"to": "boss@company.com", "subject": "项目进展"},
        risk_level="high",
    )
    print(execute_tool(dangerous_request))
    print(execute_tool(dangerous_request, approved=True))

    injection_request = ToolRequest(
        tool_name="calculate",
        args={"expression": "__import__('os').system('dir')"},
        risk_level="low",
    )
    try:
        print(execute_tool(injection_request))
    except ValueError as exc:
        print(f"恶意表达式已拒绝：{exc}")


if __name__ == "__main__":
    main()

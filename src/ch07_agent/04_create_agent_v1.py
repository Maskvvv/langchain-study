"""
第七章：Agent — LangChain v1 create_agent

本脚本演示现代 LangChain Agent 的推荐入口 create_agent()。

重点：
1. create_agent() 返回的是基于 LangGraph 的可执行图
2. 可以直接绑定工具
3. 可以接入 checkpointer、interrupt、response_format、middleware 等工程能力

运行方式：
    uv run python src/ch07_agent/04_create_agent_v1.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气。只支持北京、上海、深圳。"""
    weather_data = {
        "北京": "晴天，25°C，微风",
        "上海": "多云，28°C，东南风3级",
        "深圳": "阵雨，30°C，南风2级",
    }
    return weather_data.get(city, f"暂无{city}的天气信息")


@tool
def calculate(expression: str) -> str:
    """计算简单数学表达式。输入示例：'123 * 456'。"""
    # 教学示例里保持简单。生产代码请使用第 11 章的安全计算器写法。
    allowed = set("0123456789+-*/.() ")
    if not all(char in allowed for char in expression):
        return "错误：表达式包含不允许的字符"
    return f"{expression} = {eval(expression)}"  # noqa: S307


def build_agent():
    """创建一个现代 LangChain Agent。

    这里只构建 Agent，不主动调用模型，方便在没有网络时也能查看结构。
    """
    model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    return create_agent(
        model=model,
        tools=[search_weather, calculate],
        system_prompt=(
            "你是一个可靠的企业助手。"
            "回答用户问题前，先判断是否需要调用工具；"
            "如果工具结果不足以回答，要明确说明不知道。"
        ),
    )


def main() -> None:
    print("=" * 60)
    print("第七章：LangChain v1 create_agent")
    print("=" * 60)

    agent = build_agent()

    print("Agent 已创建。它本质上是一个可执行的 LangGraph：")
    print(f"  类型: {type(agent).__name__}")
    print(f"  节点: {list(agent.nodes.keys())}")

    print("\n真实调用示例：")
    print('  result = agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})')

    print("\n学习重点：")
    print("  1. create_agent 适合快速搭建现代工具调用 Agent")
    print("  2. 复杂企业流程仍建议手写 LangGraph，显式控制每个节点")


if __name__ == "__main__":
    main()

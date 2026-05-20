"""
第六章：Tool & Function Calling — 多工具协作

本脚本演示 LangChain 中多工具协作的用法，包括：
1. 多工具绑定与自动选择
2. 工具调用的错误处理
3. 构建一个多工具助手

运行方式：
    uv run python src/ch06_tool_function_calling/03_multi_tool_collaboration.py
"""

import json
from datetime import datetime

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_multi_tool_assistant():
    """演示多工具协作助手"""

    # ========================================
    # 定义多个工具
    # ========================================
    # 每个工具负责一个特定领域
    # LLM 会根据用户问题自动选择合适的工具

    @tool
    def get_current_time() -> str:
        """获取当前的日期和时间"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        weather_data = {
            "北京": "晴天，25°C，微风，空气质量良好",
            "上海": "多云，28°C，东南风3级，湿度65%",
            "深圳": "阵雨，30°C，南风2级，湿度80%",
            "成都": "阴天，22°C，微风，空气质量一般",
        }
        return weather_data.get(city, f"暂无{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果。输入必须是合法的数学表达式，如 '2+3*4'"""
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def search_knowledge(query: str) -> str:
        """在知识库中搜索信息。可以搜索技术文档、百科知识等。"""
        knowledge_base = {
            "LangChain": "LangChain 是一个用于构建 LLM 应用的框架，核心模块包括 Model I/O、Retrieval、Chains、Memory 和 Agents。",
            "RAG": "RAG（检索增强生成）是一种将外部知识检索与 LLM 生成结合的技术，能有效减少幻觉。",
            "Python": "Python 是一种解释型编程语言，以简洁优雅著称，广泛应用于数据科学、AI、Web 开发等领域。",
        }
        for key, value in knowledge_base.items():
            if key.lower() in query.lower():
                return value
        return f"未找到与 '{query}' 相关的信息"

    # 工具映射
    tools = [get_current_time, search_weather, calculate, search_knowledge]
    tools_map = {t.name: t for t in tools}

    # 绑定工具到模型
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    llm_with_tools = llm.bind_tools(tools)

    # ========================================
    # 多轮对话 + 工具调用
    # ========================================
    # 这是一个完整的 Agent 循环：
    # 1. LLM 决定是否调用工具
    # 2. 如果需要，执行工具
    # 3. 将结果返回给 LLM
    # 4. LLM 可能继续调用工具或生成最终回答
    # 5. 重复直到 LLM 不再调用工具

    def agent_loop(question: str, max_iterations: int = 5) -> str:
        """Agent 循环：处理工具调用直到获得最终回答"""
        messages = [HumanMessage(content=question)]

        for i in range(max_iterations):
            # 调用 LLM
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # 如果没有工具调用，返回最终回答
            if not response.tool_calls:
                return response.content

            # 执行工具调用
            print(f"\n  🔄 第 {i + 1} 轮工具调用:")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"    🔧 调用: {tool_name}({tool_args})")

                # 执行工具
                tool_func = tools_map[tool_name]
                try:
                    tool_result = tool_func.invoke(tool_args)
                except Exception as e:
                    tool_result = f"工具执行错误: {e}"

                print(f"    📋 结果: {tool_result}")

                # 将结果追加到消息列表
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                )

        return "达到最大迭代次数，未能获得最终回答"

    # 测试不同类型的问题
    questions = [
        "现在几点了？",
        "北京和上海今天天气怎么样？",
        "帮我算一下 (123 + 456) * 789",
        "什么是 RAG？",
        "现在几点了？北京天气怎么样？帮我算一下 100 * 200",
    ]

    for question in questions:
        print(f"\n{'=' * 50}")
        print(f"❓ 问题: {question}")
        answer = agent_loop(question)
        print(f"\n💡 回答: {answer}")


def demo_tool_error_handling():
    """演示工具调用的错误处理"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    @tool
    def divide(a: float, b: float) -> str:
        """计算 a 除以 b 的结果"""
        if b == 0:
            return "错误：除数不能为零"
        return f"{a} / {b} = {a / b}"

    llm_with_tools = llm.bind_tools([divide])
    tools_map = {"divide": divide}

    # ========================================
    # 错误处理：工具返回错误信息
    # ========================================
    # 当工具执行出错时，将错误信息作为 ToolMessage 返回
    # LLM 可以理解错误并给出合理的回答

    question = "请计算 10 除以 0"
    messages = [HumanMessage(content=question)]
    print(f"\n❓ 问题: {question}")

    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    for tool_call in ai_response.tool_calls:
        tool_result = tools_map[tool_call["name"]].invoke(tool_call["args"])
        print(f"  🔧 工具结果: {tool_result}")
        messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )

    final_response = llm_with_tools.invoke(messages)
    print(f"  💡 最终回答: {final_response.content}")
    # LLM 能理解"除数不能为零"的错误并给出合理解释 ✅


def main():
    print("=" * 60)
    print("📚 多工具协作演示")
    print("=" * 60)

    demo_multi_tool_assistant()
    demo_tool_error_handling()

    print("\n✅ 多工具协作演示完成！")


if __name__ == "__main__":
    main()

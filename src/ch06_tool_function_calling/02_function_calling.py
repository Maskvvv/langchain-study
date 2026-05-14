"""
第六章：Tool & Function Calling — Function Calling 实战

本脚本演示 LangChain 中 Function Calling 的完整流程，包括：
1. 将工具绑定到模型
2. 处理模型的工具调用请求
3. 将工具结果返回给模型
4. 结构化输出

运行方式：
    uv run python src/ch06_tool_function_calling/02_function_calling.py
"""

import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


def demo_bind_tools():
    """演示将工具绑定到模型"""
    # ========================================
    # bind_tools()：将工具定义附加到模型
    # ========================================
    # 调用 bind_tools() 后，模型就知道有哪些工具可用
    # 模型会根据用户问题决定是否调用工具

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        weather_data = {
            "北京": "晴天，25°C，微风",
            "上海": "多云，28°C，东南风3级",
            "深圳": "阵雨，30°C，南风2级",
        }
        return weather_data.get(city, f"未找到{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果"""
        try:
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # 绑定工具
    llm_with_tools = llm.bind_tools([search_weather, calculate])

    # ========================================
    # 场景1：不需要工具的问题
    # ========================================
    response = llm_with_tools.invoke("你好，请介绍一下自己")
    print(f"不需要工具的问题:")
    print(f"  回复: {response.content}")
    print(f"  工具调用: {response.tool_calls}")
    # tool_calls 为空列表，说明模型认为不需要调用工具

    # ========================================
    # 场景2：需要工具的问题
    # ========================================
    response = llm_with_tools.invoke("北京今天天气怎么样？")
    print(f"\n需要工具的问题:")
    print(f"  回复: {response.content}")
    print(f"  工具调用: {response.tool_calls}")
    # tool_calls 包含模型决定调用的工具信息
    # 包括工具名、参数和调用 ID

    return llm_with_tools, [search_weather, calculate]


def demo_full_tool_calling_flow():
    """演示完整的工具调用流程"""
    # ========================================
    # Function Calling 完整流程
    # ========================================
    # 1. 用户提问
    # 2. LLM 决定调用工具，返回 tool_calls
    # 3. 应用执行工具，获取结果
    # 4. 将结果作为 ToolMessage 返回给 LLM
    # 5. LLM 基于工具结果生成最终回答

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        weather_data = {
            "北京": "晴天，25°C，微风",
            "上海": "多云，28°C，东南风3级",
            "深圳": "阵雨，30°C，南风2级",
        }
        return weather_data.get(city, f"未找到{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果"""
        try:
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # 工具映射表：工具名 → 工具函数
    tools_map = {
        "search_weather": search_weather,
        "calculate": calculate,
    }

    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)
    llm_with_tools = llm.bind_tools([search_weather, calculate])

    # 第一步：用户提问
    user_question = "北京和上海今天天气怎么样？另外算一下 123 * 456"
    messages = [HumanMessage(content=user_question)]
    print(f"用户: {user_question}")

    # 第二步：LLM 决定调用哪些工具
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    print(f"\nLLM 决定调用 {len(ai_response.tool_calls)} 个工具:")
    for tc in ai_response.tool_calls:
        print(f"  🔧 {tc['name']}({tc['args']})")

    # 第三步：执行每个工具调用
    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # 执行工具
        tool_func = tools_map[tool_name]
        tool_result = tool_func.invoke(tool_args)

        print(f"\n  结果: {tool_result}")

        # 将结果作为 ToolMessage 追加到消息列表
        # tool_call_id 必须与 AI 返回的 tool_call id 对应
        # 这样 LLM 才能将结果与对应的工具调用关联起来
        messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_id)
        )

    # 第四步：LLM 基于工具结果生成最终回答
    final_response = llm_with_tools.invoke(messages)
    print(f"\n💡 最终回答: {final_response.content}")


def demo_structured_output():
    """演示结构化输出 — with_structured_output()"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # with_structured_output()
    # ========================================
    # 这是 Function Calling 的一种简化用法
    # 不需要手动处理 tool_calls 和 ToolMessage
    # 直接将 LLM 输出解析为 Pydantic 模型
    #
    # 底层原理：使用 Function Calling，但隐藏了工具调用的细节
    # LLM 被强制以指定的 Schema 输出，然后自动解析

    class MovieReview(BaseModel):
        """电影评价"""
        movie_name: str = Field(description="电影名称")
        rating: float = Field(description="评分，1-10分")
        summary: str = Field(description="一句话评价")
        pros: list[str] = Field(description="优点列表")
        cons: list[str] = Field(description="缺点列表")

    structured_llm = llm.with_structured_output(MovieReview)

    result = structured_llm.invoke("请评价电影《盗梦空间》")
    print(f"\n🎬 结构化输出结果:")
    print(f"  电影: {result.movie_name}")
    print(f"  评分: {result.rating}/10")
    print(f"  评价: {result.summary}")
    print(f"  优点: {', '.join(result.pros)}")
    print(f"  缺点: {', '.join(result.cons)}")
    print(f"  结果类型: {type(result)}")


def main():
    print("=" * 60)
    print("📚 Function Calling 实战演示")
    print("=" * 60)

    demo_bind_tools()
    print("\n" + "=" * 60)
    demo_full_tool_calling_flow()
    print("\n" + "=" * 60)
    demo_structured_output()

    print("\n✅ Function Calling 实战演示完成！")


if __name__ == "__main__":
    main()

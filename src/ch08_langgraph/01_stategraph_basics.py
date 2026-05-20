"""
第八章：LangGraph — StateGraph 基础

本脚本演示 LangGraph 的基础用法，包括：
1. 定义 State
2. 创建节点
3. 构建和编译 StateGraph
4. 运行图

运行方式：
    uv run python src/ch08_langgraph/01_stategraph_basics.py
"""

from typing import Annotated

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


def demo_basic_stategraph():
    """演示最基本的 StateGraph"""

    # ========================================
    # 第一步：定义 State
    # ========================================
    # State 是图在执行过程中维护的数据
    # 通常使用 TypedDict 定义
    #
    # Annotated[list, add_messages] 的含义：
    # - list: 字段类型是列表
    # - add_messages: Reducer 函数，定义如何合并更新
    #   add_messages 的行为是"追加"而非"替换"
    #   即：节点返回新的消息列表，会追加到现有消息列表后面
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # ========================================
    # 第二步：定义工具
    # ========================================
    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的天气"""
        weather_data = {
            "北京": "晴天，25°C，微风",
            "上海": "多云，28°C，东南风3级",
            "深圳": "阵雨，30°C",
        }
        return weather_data.get(city, f"暂无{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    tools = [search_weather, calculate]
    tools_map = {t.name: t for t in tools}

    # ========================================
    # 第三步：创建 LLM 并绑定工具
    # ========================================
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    llm_with_tools = llm.bind_tools(tools)

    # ========================================
    # 第四步：定义节点函数
    # ========================================
    # 每个节点函数接收 State，返回 State 的更新（部分更新即可）

    def agent_node(state: State) -> dict:
        """Agent 节点：调用 LLM，决定下一步行动"""
        # 从 State 中获取消息列表
        messages = state["messages"]
        # 调用 LLM
        response = llm_with_tools.invoke(messages)
        # 返回 State 更新：将 AI 的回复追加到消息列表
        return {"messages": [response]}

    def tools_node(state: State) -> dict:
        """工具节点：执行 LLM 请求的工具调用"""
        # 获取最后一条 AI 消息中的工具调用
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        # 执行每个工具调用
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # 执行工具
            tool_func = tools_map[tool_name]
            tool_result = tool_func.invoke(tool_args)

            # 创建 ToolMessage
            tool_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_id)
            )

        # 返回 State 更新：将工具结果追加到消息列表
        return {"messages": tool_messages}

    # ========================================
    # 第五步：定义路由函数
    # ========================================
    # 条件边需要路由函数来决定下一个节点
    def should_continue(state: State) -> str:
        """判断是否需要继续调用工具"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"  # 需要调用工具 → 跳转到 tools 节点
        return END  # 不需要工具 → 结束

    # ========================================
    # 第六步：构建 StateGraph
    # ========================================
    # StateGraph 是 LangGraph 的核心类
    # 它定义了图的节点、边和状态结构

    graph = StateGraph(State)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    # 设置入口节点
    graph.set_entry_point("agent")

    # 添加条件边：agent 节点执行后，根据 should_continue 决定下一个节点
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # 添加普通边：tools 节点执行后，回到 agent 节点
    graph.add_edge("tools", "agent")

    # ========================================
    # 第七步：编译图
    # ========================================
    # 编译会验证图的正确性（如是否有孤立节点、是否有出口等）
    # 编译后返回一个可执行的 CompiledGraph
    compiled_graph = graph.compile()

    # ========================================
    # 第八步：运行图
    # ========================================
    print("=" * 50)
    print("🤖 LangGraph 基础 Agent")
    print("=" * 50)

    questions = [
        "北京今天天气怎么样？",
        "帮我算一下 123 * 456",
        "你好，介绍一下你自己",
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")

        # invoke() 运行整个图
        # 输入是初始 State（至少包含 messages）
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=question)]}
        )

        # 获取最后一条 AI 消息作为回答
        last_ai_msg = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                last_ai_msg = msg
                break

        print(f"💡 回答: {last_ai_msg.content if last_ai_msg else '无回答'}")

    return compiled_graph


def demo_graph_visualization():
    """演示图的节点信息查看"""
    print("\n" + "=" * 50)
    print("📊 图结构信息")
    print("=" * 50)

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def dummy_agent(state: State) -> dict:
        return {"messages": []}

    def dummy_tools(state: State) -> dict:
        return {"messages": []}

    graph = StateGraph(State)
    graph.add_node("agent", dummy_agent)
    graph.add_node("tools", dummy_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", lambda s: "tools" if True else END, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    # 查看图的节点
    print(f"节点列表: {list(compiled.nodes.keys())}")
    print(f"入口节点: agent")
    print(f"出口标记: END")


def main():
    print("=" * 60)
    print("📚 LangGraph StateGraph 基础演示")
    print("=" * 60)

    demo_basic_stategraph()
    demo_graph_visualization()

    print("\n✅ LangGraph StateGraph 基础演示完成！")


if __name__ == "__main__":
    main()

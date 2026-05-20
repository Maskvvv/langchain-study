"""
第八章：LangGraph — 条件路由

本脚本演示 LangGraph 中条件路由的用法，包括：
1. 基于意图的路由
2. 多分支条件路由
3. 自定义路由逻辑

运行方式：
    uv run python src/ch08_langgraph/02_conditional_routing.py
"""

from typing import Annotated, Literal

import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


def demo_intent_routing():
    """演示基于意图的条件路由"""

    # ========================================
    # 场景：根据用户意图路由到不同的处理节点
    # ========================================
    # 这是 LangGraph 条件路由最典型的应用
    # 比传统的 if-else 更灵活，因为路由逻辑可以很复杂

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        intent: str

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 意图识别节点
    # ========================================
    # 使用 LLM 识别用户意图
    class IntentResult(BaseModel):
        intent: Literal["weather", "math", "chat"] = Field(
            description="用户意图：weather=天气查询, math=数学计算, chat=普通聊天"
        )

    intent_llm = llm.with_structured_output(IntentResult)

    def classify_intent(state: State) -> dict:
        """识别用户意图"""
        last_message = state["messages"][-1]
        result = intent_llm.invoke(
            f"判断以下用户消息的意图（weather/math/chat）：\n{last_message.content}"
        )
        return {"intent": result.intent}

    # ========================================
    # 不同意图的处理节点
    # ========================================

    def weather_node(state: State) -> dict:
        """处理天气查询"""
        from langchain_core.messages import AIMessage

        last_message = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个天气助手，用简洁的方式回答天气问题。如果用户没有指定城市，请询问。"),
            last_message,
        ])
        return {"messages": [AIMessage(content=response.content)]}

    def math_node(state: State) -> dict:
        """处理数学计算"""
        from langchain_core.messages import AIMessage

        last_message = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个数学助手，帮助用户计算数学问题。请给出详细的计算步骤。"),
            last_message,
        ])
        return {"messages": [AIMessage(content=response.content)]}

    def chat_node(state: State) -> dict:
        """处理普通聊天"""
        from langchain_core.messages import AIMessage

        last_message = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个友好的聊天助手。"),
            last_message,
        ])
        return {"messages": [AIMessage(content=response.content)]}

    # ========================================
    # 路由函数
    # ========================================
    def route_by_intent(state: State) -> str:
        """根据意图路由到不同的处理节点"""
        intent = state["intent"]
        routing_map = {
            "weather": "weather",
            "math": "math",
            "chat": "chat",
        }
        return routing_map.get(intent, "chat")

    # ========================================
    # 构建图
    # ========================================
    graph = StateGraph(State)

    # 添加节点
    graph.add_node("classify", classify_intent)
    graph.add_node("weather", weather_node)
    graph.add_node("math", math_node)
    graph.add_node("chat", chat_node)

    # 设置入口
    graph.set_entry_point("classify")

    # 添加条件边：根据意图路由
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "weather": "weather",
            "math": "math",
            "chat": "chat",
        },
    )

    # 所有处理节点执行后结束
    graph.add_edge("weather", END)
    graph.add_edge("math", END)
    graph.add_edge("chat", END)

    # 编译并运行
    compiled = graph.compile()

    print("=" * 50)
    print("🔀 条件路由演示")
    print("=" * 50)

    test_cases = [
        "北京今天天气怎么样？",
        "帮我算一下 123 * 456",
        "你好，今天心情不错！",
    ]

    for user_input in test_cases:
        result = compiled.invoke({
            "messages": [HumanMessage(content=user_input)],
            "intent": "",
        })
        last_msg = result["messages"][-1]
        print(f"\n❓ 用户: {user_input}")
        print(f"🎯 意图: {result['intent']}")
        print(f"💡 回答: {last_msg.content[:100]}...")


def demo_multi_step_routing():
    """演示多步骤条件路由"""
    # ========================================
    # 场景：一个更复杂的工作流
    # 包含审核环节的条件路由
    # ========================================

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        needs_review: bool
        approved: bool

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    def generate_response(state: State) -> dict:
        """生成回复"""
        from langchain_core.messages import AIMessage

        last_message = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个助手。如果回复涉及敏感信息（如金额、密码等），标记为需要审核。"),
            last_message,
        ])

        # 简单判断是否需要审核
        needs_review = any(
            keyword in response.content
            for keyword in ["金额", "密码", "账号", "转账"]
        )

        return {
            "messages": [AIMessage(content=response.content)],
            "needs_review": needs_review,
        }

    def review_node(state: State) -> dict:
        """审核节点（模拟人工审核）"""
        from langchain_core.messages import AIMessage

        # 在实际应用中，这里会暂停等待人工审核
        # 现在模拟自动通过
        return {
            "approved": True,
            "messages": [AIMessage(content="[审核通过]")],
        }

    def route_after_generation(state: State) -> str:
        """生成回复后，判断是否需要审核"""
        if state["needs_review"]:
            return "review"
        return END

    def route_after_review(state: State) -> str:
        """审核后，判断是否通过"""
        if state["approved"]:
            return END
        return "generate"  # 未通过，重新生成

    graph = StateGraph(State)
    graph.add_node("generate", generate_response)
    graph.add_node("review", review_node)

    graph.set_entry_point("generate")
    graph.add_conditional_edges("generate", route_after_generation, {
        "review": "review",
        END: END,
    })
    graph.add_conditional_edges("review", route_after_review, {
        "generate": "generate",
        END: END,
    })

    compiled = graph.compile()

    print("\n" + "=" * 50)
    print("🔀 多步骤条件路由演示")
    print("=" * 50)

    test_cases = [
        "今天天气怎么样？",
        "请告诉我转账金额",
    ]

    for user_input in test_cases:
        result = compiled.invoke({
            "messages": [HumanMessage(content=user_input)],
            "needs_review": False,
            "approved": False,
        })
        print(f"\n❓ 用户: {user_input}")
        print(f"📋 需要审核: {result['needs_review']}")
        print(f"✅ 审核通过: {result['approved']}")


def main():
    print("=" * 60)
    print("📚 LangGraph 条件路由演示")
    print("=" * 60)

    demo_intent_routing()
    demo_multi_step_routing()

    print("\n✅ LangGraph 条件路由演示完成！")


if __name__ == "__main__":
    main()

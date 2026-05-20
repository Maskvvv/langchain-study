"""
第八章：LangGraph — Human-in-the-Loop 人机协作

本脚本演示 LangGraph 中人机协作的用法，包括：
1. interrupt_before / interrupt_after 暂停执行
2. 人工审核与确认
3. 修改和恢复执行

运行方式：
    uv run python src/ch08_langgraph/03_human_in_the_loop.py
"""

from typing import Annotated

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


def demo_interrupt_before():
    """演示 interrupt_before — 在节点执行前暂停"""

    # ========================================
    # Human-in-the-Loop 的原理
    # ========================================
    # 在关键节点前/后暂停图的执行
    # 等待人类确认、修改或提供额外信息
    # 然后继续执行
    #
    # 典型应用场景：
    # 1. 执行不可逆操作前确认（如发送邮件、删除数据）
    # 2. 审核 AI 生成的内容
    # 3. 让人类提供 AI 无法获取的信息

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """发送邮件"""
        return f"邮件已发送给 {to}，主题: {subject}"

    @tool
    def search_info(query: str) -> str:
        """搜索信息"""
        return f"关于'{query}'的搜索结果：这是模拟的搜索结果"

    tools = [send_email, search_info]
    tools_map = {t.name: t for t in tools}

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: State) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_func = tools_map[tool_call["name"]]
            tool_result = tool_func.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        return {"messages": tool_messages}

    def should_continue(state: State) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            # 检查是否包含需要人工确认的工具
            for tc in last_message.tool_calls:
                if tc["name"] == "send_email":
                    return "human_review"
            return "tools"
        return END

    def human_review_node(state: State) -> dict:
        """人工审核节点（模拟）"""
        # 在实际应用中，这里会暂停等待人工输入
        # 现在模拟自动批准
        last_message = state["messages"][-1]
        print("\n⚠️ 需要人工审核以下操作：")
        for tc in last_message.tool_calls:
            if tc["name"] == "send_email":
                print(f"  📧 发送邮件给: {tc['args'].get('to')}")
                print(f"  📋 主题: {tc['args'].get('subject')}")
                print(f"  📝 内容: {tc['args'].get('body', '')[:50]}...")
                print("  ✅ 审核结果: 批准")

        # 执行工具
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_func = tools_map[tool_call["name"]]
            tool_result = tool_func.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        return {"messages": tool_messages}

    # 构建图
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "human_review": "human_review",
        END: END,
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", "agent")

    # 使用 interrupt_before 在 human_review 节点前暂停
    # 这样人工可以在工具执行前审核和修改
    compiled = graph.compile(interrupt_before=["human_review"])

    print("=" * 50)
    print("🧑‍💻 Human-in-the-Loop 演示")
    print("=" * 50)

    # 模拟：用户请求发送邮件
    user_input = "请帮我发一封邮件给 boss@company.com，主题是'项目进展'，内容是'项目进展顺利，预计下周完成'"
    print(f"\n❓ 用户: {user_input}")

    # 第一次调用：图会在 human_review 节点前暂停
    result = compiled.invoke({"messages": [HumanMessage(content=user_input)]})

    # 检查是否暂停
    # 在实际应用中，这里会将暂停状态保存，等待人工操作后恢复
    print(f"\n📊 图执行状态: 暂停在 human_review 节点前")
    print(f"📋 待审核的工具调用:")
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            print(f"  🔧 {tc['name']}({tc['args']})")

    # 模拟人工确认后继续执行
    # 在实际应用中，这里可以修改工具参数后再继续
    print("\n▶️ 人工确认后继续执行...")
    result = compiled.invoke(None, config={"configurable": {"thread_id": "1"}})

    # 获取最终回答
    if result and "messages" in result:
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"💡 最终回答: {msg.content[:100]}")
                break


def demo_approval_workflow():
    """演示完整的审批工作流"""

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        action_type: str
        approved: bool

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    def classify_action(state: State) -> dict:
        """分类用户请求的操作类型"""
        from pydantic import BaseModel, Field

        class ActionClassify(BaseModel):
            action_type: str = Field(description="操作类型: safe/dangerous")

        classifier = llm.with_structured_output(ActionClassify)
        last_msg = state["messages"][-1]
        result = classifier.invoke(
            f"判断以下请求是否涉及危险操作（如删除、发送、修改重要数据等）：\n{last_msg.content}"
        )
        return {"action_type": result.action_type}

    def execute_safe(state: State) -> dict:
        """执行安全操作"""
        last_msg = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个助手，直接帮助用户完成请求。"),
            last_msg,
        ])
        return {"messages": [AIMessage(content=response.content)], "approved": True}

    def request_approval(state: State) -> dict:
        """请求审批（模拟）"""
        print("\n⚠️ 此操作需要审批！")
        print("  ✅ 审批结果: 通过（模拟）")
        return {"approved": True}

    def execute_dangerous(state: State) -> dict:
        """执行危险操作（审批后）"""
        last_msg = state["messages"][-1]
        response = llm.invoke([
            SystemMessage(content="你是一个助手，用户已批准此操作，请执行。"),
            last_msg,
        ])
        return {"messages": [AIMessage(content=response.content)]}

    def route_by_action_type(state: State) -> str:
        if state["action_type"] == "dangerous":
            return "request_approval"
        return "execute_safe"

    def route_after_approval(state: State) -> str:
        if state["approved"]:
            return "execute_dangerous"
        return END

    graph = StateGraph(State)
    graph.add_node("classify", classify_action)
    graph.add_node("execute_safe", execute_safe)
    graph.add_node("request_approval", request_approval)
    graph.add_node("execute_dangerous", execute_dangerous)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_by_action_type, {
        "request_approval": "request_approval",
        "execute_safe": "execute_safe",
    })
    graph.add_edge("execute_safe", END)
    graph.add_conditional_edges("request_approval", route_after_approval, {
        "execute_dangerous": "execute_dangerous",
        END: END,
    })
    graph.add_edge("execute_dangerous", END)

    compiled = graph.compile()

    print("\n" + "=" * 50)
    print("📋 审批工作流演示")
    print("=" * 50)

    test_cases = [
        "帮我查一下明天的天气",
        "请删除所有测试数据",
    ]

    for user_input in test_cases:
        result = compiled.invoke({
            "messages": [HumanMessage(content=user_input)],
            "action_type": "",
            "approved": False,
        })
        print(f"\n❓ 用户: {user_input}")
        print(f"🎯 操作类型: {result['action_type']}")
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"💡 回答: {last_msg.content[:100]}")


def main():
    print("=" * 60)
    print("📚 LangGraph Human-in-the-Loop 演示")
    print("=" * 60)

    demo_interrupt_before()
    demo_approval_workflow()

    print("\n✅ LangGraph Human-in-the-Loop 演示完成！")
    print("\n💡 Human-in-the-Loop 的关键点：")
    print("  1. 使用 interrupt_before/interrupt_after 在关键节点暂停")
    print("  2. 暂停后可以审核、修改 State")
    print("  3. 确认后使用 invoke(None) 继续执行")
    print("  4. 适用于所有需要人工确认的场景")


if __name__ == "__main__":
    main()

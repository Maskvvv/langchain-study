"""
第九章：企业级 LangGraph — Checkpoint 与 thread_id

本脚本演示 LangGraph 如何通过 checkpointer 保存同一个 thread_id 的状态。
为了避免依赖真实模型，本示例使用确定性 Python 函数模拟 Agent 节点。

运行方式：
    uv run python src/ch09_production_langgraph/01_checkpoint_thread.py
"""

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """图状态。

    messages 使用 add_messages reducer，表示新消息会追加到历史消息后面。
    turn_count 是普通字段，节点返回的新值会覆盖旧值。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def assistant_node(state: State) -> dict:
    """模拟一个助手节点。

    真实项目中，这里通常会调用 llm.invoke()。
    本示例故意不调模型，让你专注理解 checkpoint 的状态恢复。
    """
    turn_count = state.get("turn_count", 0) + 1
    last_human = next(
        msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
    )

    reply = (
        f"这是同一个 thread 的第 {turn_count} 轮。"
        f"我收到了你的新问题：{last_human.content}"
    )
    return {
        "messages": [AIMessage(content=reply)],
        "turn_count": turn_count,
    }


def build_graph():
    """构建带 checkpoint 的图。"""
    graph = StateGraph(State)
    graph.add_node("assistant", assistant_node)
    graph.set_entry_point("assistant")
    graph.add_edge("assistant", END)

    # InMemorySaver 适合学习和本地调试。
    # 生产环境应换成 SQLite、Postgres、Redis 等持久化 checkpointer。
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    print("=" * 60)
    print("第九章：Checkpoint 与 thread_id")
    print("=" * 60)

    app = build_graph()

    # 同一个 thread_id 表示同一条会话/任务线程。
    config = {"configurable": {"thread_id": "user-001"}}

    first = app.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明。")], "turn_count": 0},
        config=config,
    )
    print(f"第一轮：{first['messages'][-1].content}")

    second = app.invoke(
        {"messages": [HumanMessage(content="你还记得这是第几轮吗？")]},
        config=config,
    )
    print(f"第二轮：{second['messages'][-1].content}")

    print(f"\n当前 thread 中累计消息数：{len(second['messages'])}")
    print("核心结论：thread_id + checkpointer 是多轮 Agent 和断点续跑的基础。")


if __name__ == "__main__":
    main()

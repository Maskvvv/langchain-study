"""
第八章：LangGraph — Multi-Agent 多智能体编排

本脚本演示 LangGraph 中多 Agent 编排的用法，包括：
1. Supervisor 模式：中心路由 + 专门 Agent
2. 多 Agent 协作完成复杂任务

运行方式：
    uv run python src/ch08_langgraph/04_multi_agent.py
"""

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


def demo_supervisor_pattern():
    """演示 Supervisor 模式的多 Agent 编排"""

    # ========================================
    # Supervisor 模式
    # ========================================
    # 一个中心 Agent（Supervisor）负责：
    # 1. 理解用户请求
    # 2. 决定将任务分配给哪个专门 Agent
    # 3. 汇总各 Agent 的结果
    # 4. 生成最终回答
    #
    # 结构：
    #   ┌────────────┐
    #   │ Supervisor │ ← 路由器
    #   └──┬───┬───┬┘
    #      │   │   │
    #  ┌───▼┐ ┌▼──┐ ┌▼──────┐
    #  │研究 │ │写手│ │审核员 │
    #  └───┬┘ └┬──┘ └┬──────┘
    #      │   │     │
    #      └───┴─────┘
    #          │
    #     ┌────▼────┐
    #     │   END   │
    #     └─────────┘

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        next_agent: str

    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # Supervisor 节点
    # ========================================
    def supervisor_node(state: State) -> dict:
        """Supervisor：分析请求，决定分配给哪个 Agent"""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel, Field

        class SupervisorDecision(BaseModel):
            next_agent: Literal["researcher", "writer", "reviewer", "FINISH"] = Field(
                description="下一个要调用的 Agent，或 FINISH 表示完成"
            )

        supervisor_llm = llm.with_structured_output(SupervisorDecision)

        system_prompt = """你是一个任务调度器。根据用户的请求和当前对话状态，决定下一步应该由哪个 Agent 处理：

- researcher（研究员）：负责搜索信息、收集数据、回答事实性问题
- writer（写手）：负责撰写文章、创作内容、整理文字
- reviewer（审核员）：负责审核内容、检查质量、提出修改建议
- FINISH：任务已完成，生成最终回答

如果用户的问题已经得到充分回答，选择 FINISH。"""

        messages = state["messages"]
        decision = supervisor_llm.invoke([
            SystemMessage(content=system_prompt),
            *messages,
        ])

        if decision.next_agent == "FINISH":
            # 生成最终回答
            response = llm.invoke([
                SystemMessage(content="请根据以上对话内容，给用户一个完整的最终回答。"),
                *messages,
            ])
            return {
                "messages": [AIMessage(content=response.content)],
                "next_agent": "FINISH",
            }

        return {"next_agent": decision.next_agent}

    # ========================================
    # 专门 Agent 节点
    # ========================================
    def researcher_node(state: State) -> dict:
        """研究员 Agent：搜索和收集信息"""
        from langchain_core.messages import AIMessage

        response = llm.invoke([
            SystemMessage(content=(
                "你是一个资深研究员。你的任务是：\n"
                "1. 搜索和收集相关信息\n"
                "2. 提供准确的事实和数据\n"
                "3. 引用信息来源\n"
                "请用专业但易懂的方式回答。"
            )),
            *state["messages"],
        ])
        return {
            "messages": [AIMessage(content=f"[研究员] {response.content}")],
            "next_agent": "supervisor",
        }

    def writer_node(state: State) -> dict:
        """写手 Agent：撰写和创作内容"""
        from langchain_core.messages import AIMessage

        response = llm.invoke([
            SystemMessage(content=(
                "你是一个优秀的写手。你的任务是：\n"
                "1. 根据提供的信息撰写文章\n"
                "2. 确保内容流畅、有逻辑\n"
                "3. 使用生动有趣的语言\n"
                "请写出高质量的内容。"
            )),
            *state["messages"],
        ])
        return {
            "messages": [AIMessage(content=f"[写手] {response.content}")],
            "next_agent": "supervisor",
        }

    def reviewer_node(state: State) -> dict:
        """审核员 Agent：审核和改进内容"""
        from langchain_core.messages import AIMessage

        response = llm.invoke([
            SystemMessage(content=(
                "你是一个严格的审核员。你的任务是：\n"
                "1. 检查内容的准确性和完整性\n"
                "2. 指出需要改进的地方\n"
                "3. 给出修改建议\n"
                "请客观公正地审核。"
            )),
            *state["messages"],
        ])
        return {
            "messages": [AIMessage(content=f"[审核员] {response.content}")],
            "next_agent": "supervisor",
        }

    # ========================================
    # 路由函数
    # ========================================
    def route_from_supervisor(state: State) -> str:
        next_agent = state["next_agent"]
        if next_agent == "FINISH":
            return END
        return next_agent

    # ========================================
    # 构建图
    # ========================================
    graph = StateGraph(State)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("supervisor")

    # Supervisor 的条件路由
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # 所有 Agent 执行后回到 Supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    compiled = graph.compile()

    # ========================================
    # 运行多 Agent 系统
    # ========================================
    print("=" * 50)
    print("👥 Multi-Agent Supervisor 模式演示")
    print("=" * 50)

    user_input = "请帮我写一篇关于 LangChain 的技术介绍文章，需要先研究 LangChain 的核心特性，然后撰写文章，最后审核质量"
    print(f"\n❓ 用户: {user_input}")

    result = compiled.invoke({
        "messages": [HumanMessage(content=user_input)],
        "next_agent": "",
    })

    # 打印执行过程
    print("\n📜 执行过程：")
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            prefix = ""
            if isinstance(msg, HumanMessage):
                prefix = "👤 "
            elif "[研究员]" in msg.content:
                prefix = "🔬 "
            elif "[写手]" in msg.content:
                prefix = "✍️ "
            elif "[审核员]" in msg.content:
                prefix = "🔍 "
            else:
                prefix = "🤖 "
            content = msg.content[:150]
            print(f"  {prefix}{content}{'...' if len(msg.content) > 150 else ''}")


def demo_multi_agent_summary():
    """多 Agent 模式总结"""
    print("\n" + "=" * 50)
    print("📊 Multi-Agent 编排模式总结")
    print("=" * 50)

    print("""
┌─────────────────────────────────────────────────────┐
│              Multi-Agent 编排模式                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Supervisor 模式（本脚本演示）                    │
│     中心路由 + 专门 Agent                            │
│     适合：任务类型明确、需要协调的场景                │
│                                                     │
│  2. 对话模式                                        │
│     Agent 之间通过对话协商                           │
│     适合：需要讨论和共识的场景                       │
│                                                     │
│  3. 层级模式                                        │
│     多层 Supervisor 嵌套                            │
│     适合：非常复杂的大规模任务                       │
│                                                     │
│  4. 自定义模式                                      │
│     用 LangGraph 自由定义任意拓扑结构                │
│     适合：特殊需求                                   │
│                                                     │
│  💡 LangGraph 的优势：                              │
│  - 可以实现任何 Agent 编排模式                       │
│  - 精细控制每一步                                    │
│  - 支持人机协作                                      │
│  - 支持持久化和恢复                                  │
│  - 可视化工作流                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
""")


def main():
    print("=" * 60)
    print("📚 LangGraph Multi-Agent 演示")
    print("=" * 60)

    demo_supervisor_pattern()
    demo_multi_agent_summary()

    print("✅ LangGraph Multi-Agent 演示完成！")
    print("\n🎉 恭喜！你已经完成了整个 LangChain 学习路线！")
    print("   从基础的 Model I/O 到高级的 Multi-Agent 编排，")
    print("   你已经掌握了 LangChain 的核心知识！")


if __name__ == "__main__":
    main()

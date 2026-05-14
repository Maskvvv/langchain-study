"""
第八章：LangGraph — interrupt + checkpoint 恢复执行

本脚本演示更接近生产环境的人机协作：
1. 图执行到 review 节点时调用 interrupt() 暂停
2. checkpointer 保存当前 thread 状态
3. 人工审批后使用 Command(resume=...) 恢复执行

运行方式：
    uv run python src/ch08_langgraph/05_interrupt_resume.py
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class EmailState(TypedDict):
    """邮件审批流程的状态。"""

    user_request: str
    draft_email: str
    approved: bool
    final_result: str


def draft_email_node(state: EmailState) -> dict:
    """生成邮件草稿。

    真实项目中这里可能调用 LLM。教学示例用固定文本，避免依赖网络。
    """
    draft = (
        "收件人：boss@company.com\n"
        "主题：项目进展\n"
        "正文：项目进展顺利，预计下周完成。"
    )
    return {"draft_email": draft}


def review_node(state: EmailState) -> dict:
    """人工审批节点。

    interrupt() 会暂停图执行，并把 value 返回给调用方。
    当外部用 Command(resume=...) 恢复时，interrupt() 的返回值就是 resume 的内容。
    """
    decision = interrupt(
        {
            "type": "email_approval",
            "question": "是否批准发送这封邮件？",
            "draft_email": state["draft_email"],
        }
    )

    return {
        "approved": decision.get("approved", False),
        "draft_email": decision.get("draft_email", state["draft_email"]),
    }


def send_email_node(state: EmailState) -> dict:
    """执行发送动作。

    真实项目中这里会调用邮件 API。高风险动作必须在审批通过后执行。
    """
    if not state["approved"]:
        return {"final_result": "邮件未获批准，已取消发送。"}

    return {"final_result": f"邮件已发送：\n{state['draft_email']}"}


def build_graph():
    """构建支持暂停和恢复的图。"""
    graph = StateGraph(EmailState)
    graph.add_node("draft_email", draft_email_node)
    graph.add_node("review", review_node)
    graph.add_node("send_email", send_email_node)

    graph.set_entry_point("draft_email")
    graph.add_edge("draft_email", "review")
    graph.add_edge("review", "send_email")
    graph.add_edge("send_email", END)

    # InMemorySaver 只适合学习和本地调试。
    # 生产环境需要替换为数据库型 checkpointer。
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    print("=" * 60)
    print("第八章：interrupt + checkpoint 恢复执行")
    print("=" * 60)

    app = build_graph()
    config = {"configurable": {"thread_id": "email-review-001"}}

    first_result = app.invoke(
        {
            "user_request": "帮我给老板发项目进展邮件",
            "draft_email": "",
            "approved": False,
            "final_result": "",
        },
        config=config,
    )

    interrupt_info = first_result["__interrupt__"][0].value
    print("图已暂停，等待人工审批：")
    print(f"  问题：{interrupt_info['question']}")
    print(f"  草稿：\n{interrupt_info['draft_email']}")

    print("\n模拟人工修改并批准草稿...")
    edited_draft = interrupt_info["draft_email"].replace(
        "预计下周完成",
        "预计下周三完成，风险点已同步处理",
    )

    final_result = app.invoke(
        Command(
            resume={
                "approved": True,
                "draft_email": edited_draft,
            }
        ),
        config=config,
    )

    print("\n恢复执行后的最终结果：")
    print(final_result["final_result"])

    print("\n核心结论：interrupt() + checkpointer + thread_id 才是可跨请求恢复的人机协作基础。")


if __name__ == "__main__":
    main()

"""
第五章：Memory — 摘要记忆

本脚本演示 LangChain 中摘要记忆的用法，包括：
1. 为什么需要摘要记忆
2. 手动实现摘要记忆
3. 摘要记忆的工作原理

运行方式：
    uv run python src/ch05_memory/02_summary_memory.py
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_summary_memory():
    """演示摘要记忆的实现"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 摘要记忆的原理
    # ========================================
    # 当对话变得很长时，完整保存所有历史消息会：
    # 1. 超出 LLM 的上下文窗口
    # 2. 消耗大量 token（成本高）
    # 3. 包含很多不重要的信息
    #
    # 摘要记忆的解决方案：
    # 定期用 LLM 对历史对话生成摘要
    # 只保留摘要 + 最近几轮对话
    #
    # 这样既保留了关键信息，又控制了 token 数量

    # 模拟一段长对话
    conversation = [
        HumanMessage(content="我叫小明，我是一名Python开发者"),
        AIMessage(content="你好小明！很高兴认识一位Python开发者。有什么我可以帮你的吗？"),
        HumanMessage(content="我住在上海，在一家互联网公司工作"),
        AIMessage(content="上海是个很棒的城市！你在互联网公司做什么方向的工作呢？"),
        HumanMessage(content="我主要负责后端开发，使用Django和FastAPI"),
        AIMessage(content="Django和FastAPI都是很优秀的框架！Django适合快速开发，FastAPI则更现代和高性能。"),
        HumanMessage(content="我最近在学习LangChain，想构建AI应用"),
        AIMessage(content="LangChain是个很好的选择！它可以帮助你快速构建基于LLM的应用。"),
    ]

    # ========================================
    # 第一步：生成对话摘要
    # ========================================
    summary_prompt = ChatPromptTemplate.from_template(
        "请将以下对话内容总结为一段简洁的摘要，保留关键信息（如人名、职业、偏好等）：\n\n"
        "{conversation}"
    )

    # 将消息列表格式化为文本
    conversation_text = "\n".join(
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in conversation
    )

    summary_chain = summary_prompt | llm
    summary_response = summary_chain.invoke({"conversation": conversation_text})
    summary = summary_response.content

    print(f"📝 对话摘要:\n{summary}")
    print(f"\n原始对话 token 数（估算）: ~{len(conversation_text)} 字符")
    print(f"摘要 token 数（估算）: ~{len(summary)} 字符")
    print(f"压缩比: {len(summary) / len(conversation_text) * 100:.1f}%")

    # ========================================
    # 第二步：基于摘要 + 新问题继续对话
    # ========================================
    # 将摘要作为 SystemMessage，新问题作为 HumanMessage
    # 这样 LLM 就能基于摘要中的关键信息来回答
    context_messages = [
        SystemMessage(content=f"以下是之前对话的摘要：\n{summary}"),
        HumanMessage(content="我之前说我用什么框架做后端开发？"),
    ]

    response = llm.invoke(context_messages)
    print(f"\n基于摘要的回答: {response.content}")
    # AI 能从摘要中提取关键信息 ✅


def demo_incremental_summary():
    """演示增量摘要 — 随着对话进行不断更新摘要"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 增量摘要的原理
    # ========================================
    # 不是每次都从头总结所有对话
    # 而是在已有摘要的基础上，只总结新增的对话
    # 这样更高效，特别是对话很长的时候

    current_summary = ""
    recent_messages = []

    conversations = [
        ("我叫小明，我是一名程序员", "你好小明！"),
        ("我住在上海", "上海是个好地方！"),
        ("我正在学习LangChain", "LangChain很有趣！"),
        ("我对RAG特别感兴趣", "RAG是LangChain的核心功能之一！"),
    ]

    for user_msg, ai_msg in conversations:
        recent_messages.append(HumanMessage(content=user_msg))
        recent_messages.append(AIMessage(content=ai_msg))

        # 每两轮对话更新一次摘要
        if len(recent_messages) >= 4:
            if current_summary:
                # 增量更新：在已有摘要上追加新信息
                update_prompt = ChatPromptTemplate.from_template(
                    "以下是之前的对话摘要：\n{old_summary}\n\n"
                    "以下是对话的新增部分：\n{new_conversation}\n\n"
                    "请更新摘要，整合新增信息："
                )
                new_conv = "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in recent_messages
                )
                response = (update_prompt | llm).invoke(
                    {"old_summary": current_summary, "new_conversation": new_conv}
                )
                current_summary = response.content
            else:
                # 首次生成摘要
                conv_text = "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in recent_messages
                )
                summary_prompt = ChatPromptTemplate.from_template(
                    "请总结以下对话，保留关键信息：\n{conversation}"
                )
                response = (summary_prompt | llm).invoke({"conversation": conv_text})
                current_summary = response.content

            recent_messages = []  # 清空近期消息

    print(f"📝 最终摘要:\n{current_summary}")

    # 基于摘要回答问题
    response = llm.invoke([
        SystemMessage(content=f"对话摘要：{current_summary}"),
        HumanMessage(content="我对什么技术特别感兴趣？"),
    ])
    print(f"\n基于摘要的回答: {response.content}")


def main():
    print("=" * 60)
    print("📚 摘要记忆演示")
    print("=" * 60)

    demo_summary_memory()
    print("\n" + "=" * 60)
    demo_incremental_summary()

    print("\n✅ 摘要记忆演示完成！")


if __name__ == "__main__":
    main()

"""
第五章：Memory — 自定义记忆策略

本脚本演示如何根据需求自定义记忆策略，包括：
1. 带实体提取的记忆
2. 混合记忆策略（摘要 + 滑动窗口）
3. 记忆策略的选择指南

运行方式：
    uv run python src/ch05_memory/03_custom_memory.py
"""

from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_entity_memory():
    """演示带实体提取的记忆"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # 实体记忆的原理
    # ========================================
    # 从对话中提取关键实体（人名、地点、偏好等）
    # 单独存储实体信息，而不是保存完整对话
    #
    # 优点：
    # 1. 信息密度高，不浪费 token
    # 2. 可以精确查询特定实体
    # 3. 不会因为对话太长而丢失关键信息

    # 实体存储
    entities = {}

    # 实体提取 Prompt
    extract_prompt = ChatPromptTemplate.from_template(
        "从以下对话中提取关键实体信息，以JSON格式返回。\n"
        "格式: {{\"实体类型\": \"实体值\"}}\n"
        "例如: {{\"姓名\": \"小明\", \"职业\": \"程序员\"}}\n\n"
        "对话：\n用户: {user_msg}\nAI: {ai_msg}"
    )

    extract_chain = extract_prompt | llm

    conversations = [
        ("我叫小明，我是一名Python开发者", "你好小明！Python是一门很棒的语言。"),
        ("我住在上海浦东", "浦东是上海很有活力的区域！"),
        ("我最喜欢的框架是FastAPI", "FastAPI确实很现代和高性能！"),
    ]

    for user_msg, ai_msg in conversations:
        # 提取实体
        import json

        response = extract_chain.invoke({"user_msg": user_msg, "ai_msg": ai_msg})
        try:
            new_entities = json.loads(response.content)
            entities.update(new_entities)
        except json.JSONDecodeError:
            pass

    print(f"📋 提取的实体: {entities}")

    # 基于实体回答问题
    entity_context = "\n".join(f"- {k}: {v}" for k, v in entities.items())
    response = llm.invoke([
        SystemMessage(content=f"已知用户信息：\n{entity_context}"),
        HumanMessage(content="我住在哪里？我最喜欢什么框架？"),
    ])
    print(f"\n基于实体的回答: {response.content}")


def demo_hybrid_memory():
    """演示混合记忆策略（摘要 + 滑动窗口）"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # 混合记忆策略
    # ========================================
    # 结合摘要记忆和滑动窗口的优点：
    # - 摘要：保留历史关键信息（压缩的）
    # - 滑动窗口：保留最近几轮完整对话（精确的）
    #
    # 结构：
    # [SystemMessage: 摘要] + [最近K轮完整对话] + [当前问题]

    summary = ""
    recent_messages: List = []
    max_recent = 4  # 保留最近 4 条消息

    conversations = [
        "我叫小明，我是一名程序员",
        "我住在上海，在互联网公司工作",
        "我正在学习LangChain",
        "我对RAG技术很感兴趣",
        "LangChain的哪个模块最值得学习？",
    ]

    for user_input in conversations:
        recent_messages.append(HumanMessage(content=user_input))

        # 构建完整上下文
        context = []
        if summary:
            context.append(SystemMessage(content=f"之前对话的摘要：{summary}"))
        context.extend(recent_messages)

        response = llm.invoke(context)
        recent_messages.append(AIMessage(content=response.content))

        print(f"用户: {user_input}")
        print(f"AI: {response.content[:80]}")

        # 如果近期消息超过限制，将旧消息转为摘要
        if len(recent_messages) > max_recent:
            old_messages = recent_messages[:max_recent]
            recent_messages = recent_messages[max_recent:]

            # 生成增量摘要
            old_text = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in old_messages
            )

            if summary:
                prompt = ChatPromptTemplate.from_template(
                    "已有摘要：{old_summary}\n新增对话：\n{new_conv}\n请更新摘要："
                )
                resp = (prompt | llm).invoke(
                    {"old_summary": summary, "new_conv": old_text}
                )
                summary = resp.content
            else:
                prompt = ChatPromptTemplate.from_template(
                    "请总结以下对话，保留关键信息：\n{conversation}"
                )
                resp = (prompt | llm).invoke({"conversation": old_text})
                summary = resp.content

        print(f"  (摘要长度: {len(summary)} 字符, 近期消息: {len(recent_messages)} 条)")
        print()

    print(f"📝 最终摘要: {summary}")


def main():
    print("=" * 60)
    print("📚 自定义记忆策略演示")
    print("=" * 60)

    demo_entity_memory()
    print("\n" + "=" * 60)
    demo_hybrid_memory()

    print("\n✅ 自定义记忆策略演示完成！")
    print("\n💡 记忆策略选择指南：")
    print("  - 短对话（<10轮）：完整记忆，简单直接")
    print("  - 中等对话（10-30轮）：滑动窗口，保留最近K轮")
    print("  - 长对话（>30轮）：摘要记忆 或 混合策略")
    print("  - 需要记住特定信息：实体记忆")


if __name__ == "__main__":
    main()

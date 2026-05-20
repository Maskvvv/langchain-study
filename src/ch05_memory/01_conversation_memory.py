"""
第五章：Memory — 对话记忆基础

本脚本演示 LangChain 中对话记忆的基本用法，包括：
1. 手动管理消息历史
2. 滑动窗口记忆
3. RunnableWithMessageHistory 的使用

运行方式：
    uv run python src/ch05_memory/01_conversation_memory.py
"""

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_manual_history():
    """演示手动管理消息历史（最基础的方式）"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 手动管理消息历史
    # ========================================
    # 这是最直接的方式：自己维护一个消息列表
    # 每次调用 LLM 时传入完整历史，调用后将新消息追加到历史中
    #
    # 优点：完全可控，容易理解
    # 缺点：需要手动维护，容易出错

    chat_history = []

    # 第一轮对话
    user_input = "我叫小明，我是一名程序员"
    chat_history.append(HumanMessage(content=user_input))
    response = llm.invoke(chat_history)
    chat_history.append(response)
    print(f"用户: {user_input}")
    print(f"AI: {response.content}")

    # 第二轮对话
    user_input = "我叫什么名字？我的职业是什么？"
    chat_history.append(HumanMessage(content=user_input))
    response = llm.invoke(chat_history)
    chat_history.append(response)
    print(f"\n用户: {user_input}")
    print(f"AI: {response.content}")
    # AI 能记住之前的对话 ✅

    # 查看完整历史
    print(f"\n📜 对话历史（共 {len(chat_history)} 条消息）:")
    for msg in chat_history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:50]}...")


def demo_sliding_window():
    """演示滑动窗口记忆"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 滑动窗口记忆
    # ========================================
    # 原理：只保留最近 K 轮对话，丢弃更早的历史
    # 这是最实用的记忆策略，平衡了上下文长度和信息保留
    #
    # 为什么需要滑动窗口？
    # 1. LLM 有上下文窗口限制
    # 2. 太长的历史会增加 token 消耗和延迟
    # 3. 最近的对话通常比早期的对话更相关

    chat_history = []
    window_size = 4  # 保留最近 4 条消息（2 轮对话）

    conversations = [
        "我叫小明",
        "我住在上海",
        "我喜欢编程",
        "我正在学习 LangChain",
        "我叫什么名字？",  # 这个问题在窗口外，AI 可能不记得
    ]

    for user_input in conversations:
        chat_history.append(HumanMessage(content=user_input))
        response = llm.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content))

        print(f"用户: {user_input}")
        print(f"AI: {response.content[:80]}")

        # 滑动窗口：如果历史超过 window_size，只保留最近的
        if len(chat_history) > window_size:
            chat_history = chat_history[-window_size:]

        print(f"  (当前窗口: {len(chat_history)} 条消息)")
        print()


def demo_runnable_with_history():
    """演示 RunnableWithMessageHistory — 自动管理历史"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # RunnableWithMessageHistory
    # ========================================
    # 这是 LangChain 提供的自动历史管理方案
    # 它会自动：
    # 1. 在调用 Chain 前，从 store 中加载历史消息
    # 2. 将历史消息注入 Prompt
    # 3. 在调用后，将新消息保存到 store

    # 创建 Prompt，使用 MessagesPlaceholder 预留历史消息位置
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的助手，记住用户告诉你的信息。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    # 创建会话存储
    # key 是 session_id，value 是 InMemoryChatMessageHistory
    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # 包装 Chain，添加自动历史管理
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # ========================================
    # 使用同一个 session_id 的对话会共享历史
    # ========================================
    config = {"configurable": {"session_id": "user-1"}}

    # 第一轮
    response1 = chain_with_history.invoke(
        {"input": "我叫小明，我是一名程序员"}, config=config
    )
    print(f"第一轮 AI: {response1.content}")

    # 第二轮：AI 能记住之前的对话
    response2 = chain_with_history.invoke(
        {"input": "我叫什么名字？我的职业是什么？"}, config=config
    )
    print(f"第二轮 AI: {response2.content}")

    # ========================================
    # 不同的 session_id 有独立的历史
    # ========================================
    config2 = {"configurable": {"session_id": "user-2"}}
    response3 = chain_with_history.invoke(
        {"input": "我叫什么名字？"}, config=config2
    )
    print(f"\n不同会话 AI: {response3.content}")
    print("→ user-2 的会话没有之前的记忆 ✅")


def main():
    print("=" * 60)
    print("📚 对话记忆基础演示")
    print("=" * 60)

    demo_manual_history()
    print("\n" + "=" * 60)
    demo_sliding_window()
    print("\n" + "=" * 60)
    demo_runnable_with_history()

    print("\n✅ 对话记忆基础演示完成！")


if __name__ == "__main__":
    main()

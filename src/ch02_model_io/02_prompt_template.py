"""
第二章：Model I/O — Prompt Template 提示词模板

本脚本演示 LangChain 中 Prompt Template 的用法，包括：
1. PromptTemplate：纯文本模板
2. ChatPromptTemplate：聊天消息模板
3. MessagesPlaceholder：消息占位符
4. 模板的组合与复用

运行方式：
    uv run python src/ch02_model_io/02_prompt_template.py
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_prompt_template():
    """演示 PromptTemplate（纯文本模板）"""
    # ========================================
    # PromptTemplate：最基础的模板
    # ========================================
    # 使用 {variable_name} 作为占位符
    # 调用 format() 时传入变量值，生成最终的提示词
    template = PromptTemplate.from_template("请给我讲一个关于{topic}的笑话")

    # format() 方法将变量替换为实际值
    prompt = template.format(topic="程序员")
    print(f"格式化后的提示词: {prompt}")

    # 也可以在创建时指定 input_variables（现在通常可以自动推断）
    template2 = PromptTemplate(
        template="将以下文本从{source_lang}翻译为{target_lang}：\n{text}",
        input_variables=["source_lang", "target_lang", "text"],
    )
    prompt2 = template2.format(
        source_lang="中文", target_lang="英文", text="今天天气真好"
    )
    print(f"\n翻译模板: {prompt2}")


def demo_chat_prompt_template():
    """演示 ChatPromptTemplate（聊天消息模板）"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 方式一：from_messages() — 最常用的创建方式
    # ========================================
    # 传入一个消息元组列表，每个元组格式为 (role, content)
    # role 可以是: "system", "human", "ai", "placeholder"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}，用{style}的风格回答问题。"),
            ("human", "{question}"),
        ]
    )

    # format_messages() 返回消息列表，可以直接传给 Chat Model
    messages = prompt.format_messages(
        role="哲学家", style="深沉而富有诗意", question="人生的意义是什么"
    )
    response = llm.invoke(messages)
    print(f"哲学家风格回复: {response.content[:150]}...")

    # ========================================
    # 方式二：使用 MessagePromptTemplate 组件
    # ========================================
    # 这种方式更灵活，可以精细控制每条消息的模板
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{role}，只回答与{domain}相关的问题。"
    )
    human_template = HumanMessagePromptTemplate.from_template("{question}")

    chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
    messages = chat_prompt.format_messages(
        role="营养师", domain="健康饮食", question="每天应该喝多少水"
    )
    response = llm.invoke(messages)
    print(f"\n营养师回复: {response.content[:150]}...")


def demo_messages_placeholder():
    """演示 MessagesPlaceholder（消息占位符）"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # MessagesPlaceholder：在模板中预留消息位置
    # ========================================
    # 这在实现多轮对话时非常有用
    # 它允许我们动态插入一组消息（比如对话历史）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个友好的助手，记住用户告诉你的信息。"),
            # variable_name 指定了在 format_messages 时传入历史消息的参数名
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # ========================================
    # 模拟多轮对话
    # ========================================
    # 第一轮：没有历史消息
    from langchain_core.messages import AIMessage, HumanMessage

    chat_history = []
    messages = prompt.format_messages(
        chat_history=chat_history, input="我叫小明，我住在上海"
    )
    response = llm.invoke(messages)
    print(f"第一轮回复: {response.content}")

    # 将对话加入历史
    chat_history.extend(
        [HumanMessage(content="我叫小明，我住在上海"), AIMessage(content=response.content)]
    )

    # 第二轮：有历史消息，模型可以记住之前的对话
    messages = prompt.format_messages(
        chat_history=chat_history, input="我叫什么名字？我住在哪里？"
    )
    response = llm.invoke(messages)
    print(f"\n第二轮回复: {response.content}")


def demo_lcel_chain():
    """演示使用 LCEL 将 Prompt + Model 串联"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # LCEL（LangChain Expression Language）
    # ========================================
    # 使用管道符 | 将 Prompt Template 和 Model 串联成 Chain
    # 这是 LangChain 推荐的写法，简洁且功能强大
    #
    # 原理：Prompt Template 和 Model 都实现了 Runnable 协议
    # Runnable 协议定义了 invoke()、stream()、batch() 等方法
    # 管道符 | 的本质是调用 RunnableSequence，将前一步的输出作为后一步的输入
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}。"),
            ("human", "{question}"),
        ]
    )

    # 创建 Chain：prompt → model
    chain = prompt | llm

    # 调用 Chain 时，只需要传入模板变量
    # Chain 内部流程：
    # 1. prompt.invoke({"role": "诗人", "question": "写一首关于春天的诗"})
    #    → 生成消息列表
    # 2. llm.invoke(消息列表)
    #    → 调用模型，返回 AIMessage
    response = chain.invoke({"role": "诗人", "question": "写一首关于春天的诗"})
    print(f"LCEL Chain 回复: {response.content[:200]}...")


def main():
    print("=" * 60)
    print("📚 Prompt Template 演示")
    print("=" * 60)

    demo_prompt_template()
    demo_chat_prompt_template()
    demo_messages_placeholder()
    demo_lcel_chain()

    print("\n✅ Prompt Template 演示完成！")


if __name__ == "__main__":
    main()

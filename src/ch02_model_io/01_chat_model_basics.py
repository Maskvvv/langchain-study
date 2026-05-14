"""
第二章：Model I/O — Chat Model 基础

本脚本演示 LangChain 中 Chat Model 的基本用法，包括：
1. 创建 Chat Model 实例
2. 不同消息类型的使用
3. 批量调用
4. 响应元数据的获取

运行方式：
    uv run python src/ch02_model_io/01_chat_model_basics.py
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_invocation():
    """演示最基本的模型调用"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # 最简单的调用：直接传入字符串
    # LangChain 会自动将其包装为 HumanMessage
    response = llm.invoke("1+1等于几？")
    print(f"简单调用: {response.content}")


def demo_message_types():
    """演示不同消息类型的使用"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # SystemMessage：设定 AI 的角色和行为规则
    # ========================================
    # SystemMessage 通常放在消息列表的最前面
    # 它告诉模型"你应该以什么身份/风格来回答"
    # 这比在 HumanMessage 中写"请你扮演..."更规范、更有效
    system_msg = SystemMessage(
        content="你是一个资深 Python 开发者，回答问题时总是用代码示例来说明。"
    )

    # ========================================
    # HumanMessage：人类用户的输入
    # ========================================
    human_msg = HumanMessage(content="如何在 Python 中读取文件？")

    # ========================================
    # 多轮对话：传入消息列表
    # ========================================
    # 消息列表中的顺序就是对话的顺序
    # 模型会根据 SystemMessage 的设定来回答 HumanMessage
    response = llm.invoke([system_msg, human_msg])
    print(f"\n带 SystemMessage 的回复:\n{response.content[:200]}...")

    # ========================================
    # 多轮对话示例
    # ========================================
    # 通过传入历史消息列表，模型可以理解上下文
    messages = [
        SystemMessage(content="你是一个友好的助手。"),
        HumanMessage(content="我叫小明。"),
        AIMessage(content="你好小明！很高兴认识你！有什么我可以帮你的吗？"),
        HumanMessage(content="我叫什么名字？"),
    ]
    response = llm.invoke(messages)
    print(f"\n多轮对话回复: {response.content}")
    # 模型能记住上下文中的"小明"，因为它看到了完整的消息历史


def demo_batch_invocation():
    """演示批量调用"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # batch()：一次性发送多个请求
    # ========================================
    # 这比循环调用 invoke() 更高效，因为 LangChain 内部会做并发处理
    # 适用于需要同时处理多个独立请求的场景
    results = llm.batch(
        [
            "用一句话介绍 Python",
            "用一句话介绍 JavaScript",
            "用一句话介绍 Rust",
        ]
    )

    print("\n批量调用结果：")
    for i, result in enumerate(results):
        print(f"  [{i + 1}] {result.content}")


def demo_response_metadata():
    """演示响应元数据的获取"""
    llm = ChatOpenAI(model="kimi-k2.6", temperature=1)

    response = llm.invoke("你好")

    # ========================================
    # AIMessage 的完整结构
    # ========================================
    # content: 文本内容
    # response_metadata: 包含 token 用量、模型信息等
    # id: 消息唯一标识
    # usage_metadata: 标准化的 token 用量信息
    print("\n响应元数据：")
    print(f"  内容类型: {type(response.content)}")
    print(f"  消息 ID: {response.id}")
    print(f"  Token 用量: {response.usage_metadata}")


def main():
    print("=" * 60)
    print("📚 Chat Model 基础用法演示")
    print("=" * 60)

    demo_basic_invocation()
    demo_message_types()
    demo_batch_invocation()
    demo_response_metadata()

    print("\n✅ Chat Model 基础演示完成！")


if __name__ == "__main__":
    main()

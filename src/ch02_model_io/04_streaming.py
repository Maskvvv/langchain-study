"""
第二章：Model I/O — 流式输出

本脚本演示 LangChain 中的流式输出（Streaming），包括：
1. 基础流式输出
2. 流式输出 + Output Parser
3. 流式输出的原理

运行方式：
    uv run python src/ch02_model_io/04_streaming.py
"""

import sys

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_streaming():
    """演示基础流式输出"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 流式输出的原理
    # ========================================
    # 传统方式（invoke）：模型生成完整响应后才返回
    #   用户等待 → [..........模型思考..........] → 一次性返回完整结果
    #
    # 流式方式（stream）：模型逐 token 生成，每生成一个就立即返回
    #   用户等待 → [token1][token2][token3]...[tokenN] → 实时看到生成过程
    #
    # 底层实现：基于 Server-Sent Events (SSE) 协议
    # OpenAI API 在 stream=True 时，会以 SSE 格式逐个发送 token

    prompt = ChatPromptTemplate.from_template(
        "请写一首关于{topic}的七言绝句，要有意境。"
    )

    print("🌊 流式输出演示：")
    print("-" * 40)

    # 使用 stream() 方法获取流式输出
    # 它返回一个迭代器，每次 yield 一个 AIMessageChunk
    # AIMessageChunk 是 AIMessage 的子类，只包含一小段内容
    chain = prompt | llm | StrOutputParser()

    for chunk in chain.stream({"topic": "秋天"}):
        # 每个 chunk 是一小段文本（通常是一个 token）
        # 使用 end="" 和 flush=True 实现不换行、立即输出
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40)


def demo_streaming_with_callback():
    """演示流式输出的底层细节"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 直接对 LLM 使用 stream()
    # ========================================
    # 不经过 Output Parser，可以看到最原始的流式数据
    print("\n🔍 流式输出底层细节：")
    print("-" * 40)

    for chunk in llm.stream("用一句话介绍 Python"):
        # chunk 是 AIMessageChunk 对象
        # 它的 content 是一小段文本
        # response_metadata 只在最后一个 chunk 中有值
        print(chunk.content, end="", flush=True)

    print("\n" + "-" * 40)


def demo_invoke_vs_stream():
    """对比 invoke 和 stream 的区别"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    prompt = ChatPromptTemplate.from_template("用3句话描述{topic}")
    chain = prompt | llm | StrOutputParser()

    # ========================================
    # invoke：等待完整响应
    # ========================================
    print("\n⏳ invoke 方式（等待完整响应）：")
    result = chain.invoke({"topic": "大海"})
    print(result)

    # ========================================
    # stream：逐 token 输出
    # ========================================
    print("\n🌊 stream 方式（逐 token 输出）：")
    for chunk in chain.stream({"topic": "大海"}):
        print(chunk, end="", flush=True)
    print()

    # ========================================
    # 何时用 invoke vs stream？
    # ========================================
    # - invoke：需要完整结果后再处理（如解析 JSON、提取结构化数据）
    # - stream：需要实时展示给用户（如聊天界面、打字机效果）
    # - batch：需要同时处理多个请求（如批量翻译）


def main():
    print("=" * 60)
    print("📚 流式输出演示")
    print("=" * 60)

    demo_basic_streaming()
    demo_streaming_with_callback()
    demo_invoke_vs_stream()

    print("\n✅ 流式输出演示完成！")


if __name__ == "__main__":
    main()

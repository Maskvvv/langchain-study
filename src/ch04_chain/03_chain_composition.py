"""
第四章：Chain — 链的组合与高级模式

本脚本演示 Chain 的高级组合模式，包括：
1. 链的嵌套组合
2. with_fallbacks 容错机制
3. with_retry 重试机制
4. with_config 配置传递

运行方式：
    uv run python src/ch04_chain/03_chain_composition.py
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_fallback():
    """演示 with_fallbacks — 容错机制"""
    # ========================================
    # with_fallbacks：当一个模型失败时，自动切换到备用模型
    # ========================================
    # 这在生产环境中非常重要，因为 API 可能因为限流、网络等问题失败
    #
    # 工作原理：
    # 1. 尝试调用主模型
    # 2. 如果主模型抛出异常，自动尝试第一个备用模型
    # 3. 如果第一个备用也失败，尝试第二个，以此类推

    primary_model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    fallback_model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    parser = StrOutputParser()

    chain = prompt | primary_model | parser

    # 添加备用模型
    fallback_chain = prompt | fallback_model | parser

    chain_with_fallback = chain.with_fallbacks([fallback_chain])

    result = chain_with_fallback.invoke({"topic": "Python"})
    print(f"容错链结果: {result}")


def demo_retry():
    """演示 with_retry — 重试机制"""
    # ========================================
    # with_retry：当调用失败时自动重试
    # ========================================
    # 适用于处理临时性错误（如网络波动、API 限流）
    #
    # 参数：
    # - stop_after_attempt：最多重试次数
    # - retry_if_exception_type：只在特定异常时重试

    model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    chain = prompt | model | StrOutputParser()

    # 添加重试机制
    chain_with_retry = chain.with_retry(
        stop_after_attempt=3,
    )

    result = chain_with_retry.invoke({"topic": "LangChain"})
    print(f"重试链结果: {result}")


def demo_nested_chain():
    """演示链的嵌套组合"""
    model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 场景：先生成主题，再基于主题生成内容
    # ========================================
    # 这是一种常见的"链式思考"模式
    # 第一个 Chain 的输出作为第二个 Chain 的输入

    # Chain 1：生成一个有趣的主题
    topic_chain = (
        ChatPromptTemplate.from_template("生成一个关于{domain}的有趣主题，只输出主题名称")
        | model
        | StrOutputParser()
    )

    # Chain 2：基于主题生成故事
    story_chain = (
        ChatPromptTemplate.from_template("基于主题'{topic}'写一个简短的故事")
        | model
        | StrOutputParser()
    )

    # 嵌套组合：topic_chain 的输出 → story_chain 的输入
    # 使用 RunnablePassthrough.assign 将上一步结果映射到下一步需要的字段名
    full_chain = (
        topic_chain
        | RunnablePassthrough.assign(topic=RunnableLambda(lambda x: x))
        | story_chain
    )

    # 更简洁的写法：直接用 lambda 做字段映射
    full_chain_v2 = topic_chain | (lambda topic: story_chain.invoke({"topic": topic}))

    result = full_chain_v2.invoke({"domain": "太空探索"})
    print(f"\n嵌套链结果:\n{result[:200]}...")


def demo_map_chain():
    """演示链的 map 操作 — 对列表逐一执行"""
    model = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    chain = prompt | model | StrOutputParser()

    # ========================================
    # map()：对输入列表中的每个元素执行 Chain
    # ========================================
    # 等价于 batch()，但更灵活
    topics = ["Python", "Rust", "Go"]

    # 使用 map
    map_chain = chain.map()
    results = map_chain.invoke([{"topic": t} for t in topics])

    print("\n🗺️ map 操作结果：")
    for topic, result in zip(topics, results):
        print(f"  {topic}: {result}")


def main():
    print("=" * 60)
    print("📚 链的组合与高级模式演示")
    print("=" * 60)

    demo_fallback()
    demo_retry()
    demo_nested_chain()
    demo_map_chain()

    print("\n✅ 链的组合与高级模式演示完成！")


if __name__ == "__main__":
    main()

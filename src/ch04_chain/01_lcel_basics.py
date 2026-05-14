"""
第四章：Chain — LCEL 基础

本脚本演示 LangChain Expression Language (LCEL) 的基础用法，包括：
1. 基本管道串联
2. RunnablePassthrough 的使用
3. RunnableParallel 并行执行

运行方式：
    uv run python src/ch04_chain/01_lcel_basics.py
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_pipe():
    """演示最基本的 LCEL 管道"""
    # ========================================
    # LCEL 核心：管道符 |
    # ========================================
    # prompt | model | parser 是 LCEL 的经典三段式
    # 数据流：dict → 消息列表 → AIMessage → str
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)
    parser = StrOutputParser()

    # 使用 | 管道符串联
    chain = prompt | model | parser

    # 调用链
    result = chain.invoke({"topic": "LangChain"})
    print(f"基本管道结果: {result}")
    print(f"结果类型: {type(result)}")

    # ========================================
    # 等价的传统写法（不推荐）
    # ========================================
    # messages = prompt.format_messages(topic="LangChain")
    # ai_message = model.invoke(messages)
    # text = parser.invoke(ai_message)
    # LCEL 让这一切变得简洁！


def demo_runnable_passthrough():
    """演示 RunnablePassthrough — 传递原始输入"""
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # RunnablePassthrough：原样传递输入
    # ========================================
    # 在复杂的 Chain 中，我们经常需要：
    # 1. 将原始输入传递到下一个组件
    # 2. 同时基于原始输入计算新的字段
    # RunnablePassthrough 就是做这件事的

    prompt = ChatPromptTemplate.from_template(
        "请为'{topic}'写一句slogan"
    )

    # 场景：我们想在输出中同时保留原始的 topic 和 LLM 的回答
    chain = RunnablePassthrough.assign(
        slogan=prompt | model | StrOutputParser()
    )

    result = chain.invoke({"topic": "Python"})
    print(f"\nRunnablePassthrough 结果: {result}")
    # result 包含原始的 topic 和新增的 slogan

    # ========================================
    # 更常见的用法：在 RAG Chain 中
    # ========================================
    # RunnablePassthrough.assign(context=retriever | format_docs)
    # 保留原始的 question，同时添加检索到的 context


def demo_runnable_parallel():
    """演示 RunnableParallel — 并行执行多个 Chain"""
    model = ChatOpenAI(model="kimi-k2.6", temperature=1.7)
    parser = StrOutputParser()

    # ========================================
    # RunnableParallel：同时执行多个 Chain
    # ========================================
    # 它接收相同的输入，分别传给多个子 Chain
    # 所有子 Chain 并行执行，结果合并为一个 dict
    #
    # 原理：内部使用线程池/协程并发执行
    # 适用于：同时需要多种不同视角/风格的回答

    joke_chain = (
        ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
        | model
        | parser
    )
    poem_chain = (
        ChatPromptTemplate.from_template("写一首关于{topic}的诗")
        | model
        | parser
    )
    fact_chain = (
        ChatPromptTemplate.from_template("说一个关于{topic}的有趣事实")
        | model
        | parser
    )

    # 创建并行 Chain
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        poem=poem_chain,
        fact=fact_chain,
    )

    result = parallel.invoke({"topic": "程序员"})
    print("\n🎭 并行执行结果：")
    print(f"  笑话: {result['joke'][:100]}...")
    print(f"  诗歌: {result['poem'][:100]}...")
    print(f"  事实: {result['fact'][:100]}...")


def demo_chain_methods():
    """演示 Chain 的多种调用方式"""
    prompt = ChatPromptTemplate.from_template("用三个词描述{topic}")
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)
    chain = prompt | model | StrOutputParser()

    # ========================================
    # invoke：单次调用
    # ========================================
    result = chain.invoke({"topic": "Python"})
    print(f"\ninvoke: {result}")

    # ========================================
    # batch：批量调用
    # ========================================
    # 内部会并发执行，比循环 invoke 更高效
    results = chain.batch([{"topic": "Python"}, {"topic": "Rust"}, {"topic": "Go"}])
    print(f"\nbatch: {results}")

    # ========================================
    # stream：流式调用
    # ========================================
    print("\nstream: ", end="")
    for chunk in chain.stream({"topic": "LangChain"}):
        print(chunk, end="", flush=True)
    print()


def main():
    print("=" * 60)
    print("📚 LCEL 基础演示")
    print("=" * 60)

    demo_basic_pipe()
    demo_runnable_passthrough()
    demo_runnable_parallel()
    demo_chain_methods()

    print("\n✅ LCEL 基础演示完成！")


if __name__ == "__main__":
    main()

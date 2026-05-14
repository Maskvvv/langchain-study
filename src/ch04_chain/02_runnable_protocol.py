"""
第四章：Chain — Runnable 协议详解

本脚本深入演示 Runnable 协议的各个方面，包括：
1. 自定义 Runnable
2. RunnableLambda 和 RunnableGenerator
3. 输入输出类型的追踪

运行方式：
    uv run python src/ch04_chain/02_runnable_protocol.py
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_custom_runnable_lambda():
    """演示 RunnableLambda — 将 Python 函数包装为 Runnable"""
    # ========================================
    # RunnableLambda：将任意 Python 函数转为 Runnable
    # ========================================
    # 这是扩展 LCEL Chain 最常用的方式
    # 只需要用 RunnableLambda 包装一个函数，它就能参与管道组合

    def word_count(text: str) -> dict:
        """统计文本的词数"""
        words = text.split()
        return {
            "text": text,
            "word_count": len(words),
            "char_count": len(text),
        }

    # 包装为 Runnable
    word_count_runnable = RunnableLambda(word_count)

    # 可以在 Chain 中使用
    result = word_count_runnable.invoke("Hello world, this is a test")
    print(f"词数统计: {result}")

    # ========================================
    # 在 LCEL 管道中使用 RunnableLambda
    # ========================================
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")

    # 在 model 和 parser 之间插入自定义处理逻辑
    def add_length_info(text: str) -> str:
        return f"{text}\n\n(回复长度: {len(text)} 字符)"

    chain = (
        prompt
        | model
        | StrOutputParser()
        | RunnableLambda(add_length_info)
    )

    result = chain.invoke({"topic": "Python"})
    print(f"\n带长度信息的回复:\n{result}")


def demo_runnable_generator():
    """演示 RunnableGenerator — 流式自定义处理"""
    # ========================================
    # RunnableGenerator：在流式输出中插入自定义逻辑
    # ========================================
    # 当 Chain 使用 stream() 时，每个 chunk 会流经 Generator
    # 这允许我们实时处理流式数据

    model = ChatOpenAI(model="kimi-k2.6", temperature=1.7)
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
    chain = prompt | model | StrOutputParser()

    # 自定义流式处理器：在每个 chunk 后添加标记
    def add_chunk_index(stream):
        for i, chunk in enumerate(stream):
            if i == 0:
                yield "[开始] "
            yield chunk
            if i == 0 and chunk:
                pass
        yield " [结束]"

    print("\n🌊 流式输出 + 自定义处理：")
    for chunk in (chain | RunnableLambda(add_chunk_index)).stream({"topic": "春天"}):
        print(chunk, end="", flush=True)
    print()


def demo_input_output_schema():
    """演示 Runnable 的输入输出类型追踪"""
    prompt = ChatPromptTemplate.from_template("介绍{topic}")
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)
    chain = prompt | model | StrOutputParser()

    # ========================================
    # input_schema / output_schema
    # ========================================
    # LCEL Chain 可以自动推断输入输出类型
    # 这对于调试和理解 Chain 的数据流非常有用
    print("\n📋 Chain 的输入输出类型：")
    print(f"  输入 Schema: {chain.input_schema.schema()}")
    print(f"  输出 Schema: {chain.output_schema.schema()}")
    # input_schema 来自于管道的第一个组件（prompt）
    # output_schema 来自于管道的最后一个组件（parser）


def demo_complex_chain():
    """演示复杂 Chain 的组合"""
    model = ChatOpenAI(model="kimi-k2.6", temperature=1)

    # ========================================
    # 场景：对同一个主题，生成不同风格的内容
    # 然后汇总为一个综合报告
    # ========================================

    # 子 Chain 1：生成技术介绍
    tech_chain = (
        ChatPromptTemplate.from_template("从技术角度简要介绍{topic}")
        | model
        | StrOutputParser()
    )

    # 子 Chain 2：生成应用场景
    app_chain = (
        ChatPromptTemplate.from_template("列举{topic}的3个应用场景")
        | model
        | StrOutputParser()
    )

    # 子 Chain 3：生成优缺点
    pros_cons_chain = (
        ChatPromptTemplate.from_template("分析{topic}的优缺点")
        | model
        | StrOutputParser()
    )

    # 并行执行所有子 Chain
    parallel = RunnableParallel(
        tech=tech_chain,
        applications=app_chain,
        pros_cons=pros_cons_chain,
    )

    # 汇总函数：将并行结果格式化为报告
    def format_report(data: dict) -> str:
        return f"""
{'=' * 40}
  综合报告
{'=' * 40}

📖 技术介绍：
{data['tech']}

🎯 应用场景：
{data['applications']}

⚖️ 优缺点分析：
{data['pros_cons']}

{'=' * 40}
""".strip()

    # 完整 Chain：并行执行 → 格式化
    report_chain = parallel | RunnableLambda(format_report)

    result = report_chain.invoke({"topic": "LangChain"})
    print(result)


def main():
    print("=" * 60)
    print("📚 Runnable 协议详解")
    print("=" * 60)

    demo_custom_runnable_lambda()
    demo_runnable_generator()
    demo_input_output_schema()
    demo_complex_chain()

    print("\n✅ Runnable 协议演示完成！")


if __name__ == "__main__":
    main()

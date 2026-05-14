"""
第三章：Retrieval — Text Splitter 文本切分器

本脚本演示 LangChain 中 Text Splitter 的用法，包括：
1. RecursiveCharacterTextSplitter：递归切分（最常用）
2. 切分参数的影响
3. 切分原理详解

运行方式：
    uv run python src/ch03_retrieval/02_text_splitter.py
"""

import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def demo_recursive_splitter():
    """演示 RecursiveCharacterTextSplitter — 递归字符切分器"""
    # ========================================
    # RecursiveCharacterTextSplitter 原理
    # ========================================
    # 它是 LangChain 推荐的默认切分器，工作原理：
    # 1. 首先尝试按 "\n\n"（段落分隔符）切分
    # 2. 如果某段仍然超过 chunk_size，再按 "\n"（行分隔符）切分
    # 3. 如果某行仍然超过，再按 " "（空格/词边界）切分
    # 4. 最后按字符切分
    #
    # 这种递归策略确保了：
    # - 优先保持段落的完整性
    # - 其次保持句子的完整性
    # - 最后才按字符切分
    # - chunk_overlap 确保相邻块之间有重叠，保持上下文连贯

    with open(os.path.join(DATA_DIR, "sample.txt"), encoding="utf-8") as f:
        text = f.read()

    # 创建切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_text(text)

    print(f"📄 原文长度: {len(text)} 字符")
    print(f"📦 切分后: {len(chunks)} 个块")
    print()

    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i + 1} (长度: {len(chunk)}) ---")
        print(chunk)
        print()


def demo_chunk_parameters():
    """演示切分参数的影响"""
    # ========================================
    # chunk_size：每个块的最大大小
    # chunk_overlap：相邻块之间的重叠大小
    # ========================================
    # 选择 chunk_size 的考量：
    # - 太小：丢失上下文，语义不完整
    # - 太大：包含太多无关信息，超出 LLM 上下文窗口
    # - 推荐：500-1500 字符（取决于模型和场景）
    #
    # 选择 chunk_overlap 的考量：
    # - 太小：相邻块之间缺乏连贯性
    # - 太大：冗余信息太多，浪费 token
    # - 推荐：chunk_size 的 10%-20%

    sample_text = """
    LangChain 是一个强大的框架。它可以帮助开发者构建 AI 应用。
    LangChain 的核心模块包括 Model I/O、Retrieval、Chains、Memory 和 Agents。
    其中 Model I/O 负责与大语言模型的交互，包括提示词管理和输出解析。
    Retrieval 模块负责与外部数据的交互，包括文档加载、切分和向量检索。
    Chains 模块将多个组件串联成链式调用，实现复杂的工作流。
    Memory 模块在链调用之间持久化应用状态，实现多轮对话。
    Agents 模块根据用户输入动态决定调用哪些工具和操作。
    """.strip()

    print("🔬 不同参数的切分效果对比：\n")

    for chunk_size, chunk_overlap in [(50, 10), (100, 20), (200, 40)]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        chunks = splitter.split_text(sample_text)
        print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"  → 切分出 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            print(f"  块{i + 1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
        print()


def demo_split_documents():
    """演示切分 Document 对象（保留 metadata）"""
    from langchain_core.documents import Document

    # ========================================
    # split_documents() vs split_text()
    # ========================================
    # split_text()：只处理纯文本，返回字符串列表
    # split_documents()：处理 Document 对象，返回 Document 列表
    #   → 保留原始 metadata，并添加 start_index 等额外信息
    #   → 在 RAG 流程中必须使用 split_documents()

    docs = [
        Document(
            page_content="LangChain 是一个用于构建 LLM 应用的框架。"
            "它提供了 Model I/O、Retrieval、Chains、Memory 和 Agents 等核心模块。"
            "每个模块都是独立的组件，可以自由组合使用。",
            metadata={"source": "intro.txt", "topic": "LangChain"},
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
        separators=["。", "，", " ", ""],
    )

    split_docs = splitter.split_documents(docs)

    print("📋 切分 Document 对象（保留 metadata）：")
    for i, doc in enumerate(split_docs):
        print(f"\n  块 {i + 1}:")
        print(f"    内容: {doc.page_content}")
        print(f"    元数据: {doc.metadata}")


def main():
    print("=" * 60)
    print("📚 Text Splitter 演示")
    print("=" * 60)

    demo_recursive_splitter()
    demo_chunk_parameters()
    demo_split_documents()

    print("\n✅ Text Splitter 演示完成！")


if __name__ == "__main__":
    main()

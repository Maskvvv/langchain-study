"""
第三章：Retrieval — Document Loader 文档加载器

本脚本演示 LangChain 中 Document Loader 的用法，包括：
1. TextLoader：加载纯文本
2. CSVLoader：加载 CSV 文件
3. Document 对象的结构

运行方式：
    uv run python src/ch03_retrieval/01_document_loader.py
"""

import csv
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, TextLoader

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def demo_text_loader():
    """演示 TextLoader — 加载纯文本文件"""
    # ========================================
    # TextLoader：最简单的文档加载器
    # ========================================
    # 它读取文本文件的全部内容，封装为一个 Document 对象
    file_path = os.path.join(DATA_DIR, "sample.txt")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print(f"📄 加载了 {len(documents)} 个文档")

    # ========================================
    # Document 对象的结构
    # ========================================
    # 每个 Document 有两个核心属性：
    #   - page_content: 文档的文本内容（str）
    #   - metadata: 文档的元数据（dict），如文件路径、页码等
    doc = documents[0]
    print(f"\n文档内容（前200字）:\n{doc.page_content[:200]}")
    print(f"\n文档元数据: {doc.metadata}")


def demo_csv_loader():
    """演示 CSVLoader — 加载 CSV 文件"""
    # 先创建一个示例 CSV 文件
    csv_path = os.path.join(DATA_DIR, "sample.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category", "description"])
        writer.writerow(["Python", "编程语言", "简洁优雅的通用编程语言"])
        writer.writerow(["Rust", "编程语言", "注重安全和性能的系统编程语言"])
        writer.writerow(["Go", "编程语言", "为并发而生的编程语言"])

    # ========================================
    # CSVLoader：将 CSV 的每一行加载为一个 Document
    # ========================================
    # 每行的各列会被拼接为 page_content
    # 列名和值会以 "列名: 值" 的格式呈现
    # metadata 中包含行号 (row) 和来源 (source)
    loader = CSVLoader(csv_path, encoding="utf-8")
    documents = loader.load()

    print(f"\n📊 CSV 加载了 {len(documents)} 行数据")
    for doc in documents:
        print(f"\n  内容: {doc.page_content}")
        print(f"  元数据: {doc.metadata}")


def demo_document_structure():
    """深入理解 Document 对象"""
    from langchain_core.documents import Document

    # ========================================
    # 手动创建 Document
    # ========================================
    # Document 是 LangChain 中数据流转的基本单元
    # 在 RAG 流程中，数据始终以 Document 的形式在各组件间传递
    doc = Document(
        page_content="LangChain 是一个用于构建 LLM 应用的框架",
        metadata={
            "source": "manual",
            "author": "demo",
            "topic": "LangChain",
        },
    )

    print(f"\n📝 手动创建的 Document:")
    print(f"  内容: {doc.page_content}")
    print(f"  元数据: {doc.metadata}")

    # ========================================
    # metadata 的重要性
    # ========================================
    # metadata 在检索时非常有用：
    # 1. 可以根据 metadata 过滤文档（如只搜索某个作者的文档）
    # 2. 在生成回答时，可以引用来源信息
    # 3. 在多文档场景下，可以追踪文档的来源


def main():
    print("=" * 60)
    print("📚 Document Loader 演示")
    print("=" * 60)

    demo_text_loader()
    demo_csv_loader()
    demo_document_structure()

    print("\n✅ Document Loader 演示完成！")


if __name__ == "__main__":
    main()

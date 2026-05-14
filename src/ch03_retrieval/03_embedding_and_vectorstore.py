"""
第三章：Retrieval — Embedding 与 Vector Store

本脚本演示 LangChain 中 Embedding 和 Vector Store 的用法，包括：
1. Embedding 模型的使用
2. FAISS 向量数据库
3. 相似度检索与 MMR 检索

运行方式：
    uv run python src/ch03_retrieval/03_embedding_and_vectorstore.py
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def demo_embedding():
    """演示 Embedding 模型的使用"""
    # ========================================
    # Embedding 模型：将文本转换为向量
    # ========================================
    # OpenAI 的 text-embedding-3-small 模型
    # 输入：文本 → 输出：1536 维的浮点数向量
    #
    # 核心特性：语义相似的文本 → 向量距离更近
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 嵌入单个文本
    vector = embeddings.embed_query("猫是一种宠物")
    print(f"📐 向量维度: {len(vector)}")
    print(f"📐 向量前5维: {vector[:5]}")

    # 嵌入多个文本（批量）
    vectors = embeddings.embed_documents(
        [
            "猫是一种宠物",
            "狗是一种宠物",
            "今天股票大跌",
        ]
    )
    print(f"\n📐 批量嵌入: {len(vectors)} 个向量")

    # ========================================
    # 计算余弦相似度
    # ========================================
    # 余弦相似度衡量两个向量的方向相似程度
    # 范围 [-1, 1]，值越大表示越相似
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_1 = cosine_similarity(vectors[0], vectors[1])
    sim_2 = cosine_similarity(vectors[0], vectors[2])
    print(f"\n'猫是一种宠物' vs '狗是一种宠物' 相似度: {sim_1:.4f}")
    print(f"'猫是一种宠物' vs '今天股票大跌' 相似度: {sim_2:.4f}")
    print("→ 语义相近的文本相似度更高 ✅")


def demo_vector_store():
    """演示 Vector Store 的使用"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ========================================
    # 创建文档集合
    # ========================================
    docs = [
        Document(page_content="Python 是一种解释型编程语言，以简洁优雅著称", metadata={"topic": "Python"}),
        Document(page_content="Rust 是一种系统编程语言，注重内存安全和性能", metadata={"topic": "Rust"}),
        Document(page_content="JavaScript 是 Web 前端的核心语言，也用于后端开发", metadata={"topic": "JavaScript"}),
        Document(page_content="Go 语言由 Google 开发，擅长构建高并发服务", metadata={"topic": "Go"}),
        Document(page_content="TypeScript 是 JavaScript 的超集，添加了静态类型检查", metadata={"topic": "TypeScript"}),
    ]

    # ========================================
    # 使用 InMemoryVectorStore（内存向量数据库）
    # ========================================
    # from_documents() 会自动：
    # 1. 对每个文档的 page_content 调用 Embedding 模型生成向量
    # 2. 将文档和向量存储在内存中
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

    print("\n📦 向量数据库已创建，包含 5 个文档")

    # ========================================
    # 相似度检索
    # ========================================
    # similarity_search() 的工作流程：
    # 1. 将查询文本通过 Embedding 模型转换为向量
    # 2. 在向量数据库中计算查询向量与所有文档向量的相似度
    # 3. 返回相似度最高的 k 个文档
    results = vectorstore.similarity_search("哪种语言适合系统级编程？", k=2)
    print("\n🔍 相似度检索 '哪种语言适合系统级编程？'：")
    for doc in results:
        print(f"  → {doc.page_content} (topic: {doc.metadata['topic']})")

    # ========================================
    # 带分数的相似度检索
    # ========================================
    # similarity_search_with_score() 额外返回相似度分数
    results_with_scores = vectorstore.similarity_search_with_score(
        "Web 开发用什么语言？", k=3
    )
    print("\n🔍 带分数的检索 'Web 开发用什么语言？'：")
    for doc, score in results_with_scores:
        print(f"  → 分数: {score:.4f} | {doc.page_content}")


def demo_mmr_search():
    """演示 MMR（最大边际相关性）检索"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docs = [
        Document(page_content="Python 适合数据科学和机器学习"),
        Document(page_content="Python 在 AI 领域应用广泛"),
        Document(page_content="Python 的语法简洁易学"),
        Document(page_content="Rust 注重内存安全和零成本抽象"),
        Document(page_content="Go 适合构建微服务架构"),
    ]

    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

    # ========================================
    # MMR 检索的原理
    # ========================================
    # 普通相似度检索：只考虑与查询的相关性
    #   问题：可能返回多个内容重复的文档
    #
    # MMR 检索：同时考虑相关性和多样性
    #   公式：MMR = λ * Sim(q, d) - (1-λ) * max(Sim(d, d'))
    #   λ 越大越偏向相关性，越小越偏向多样性
    #   默认 λ = 0.5
    #
    # 效果：在保持相关性的同时，避免返回内容重复的文档

    # 对比：普通检索 vs MMR 检索
    query = "Python 的用途"

    print(f"\n🔍 查询: '{query}'\n")

    sim_results = vectorstore.similarity_search(query, k=3)
    print("普通相似度检索（可能重复）：")
    for doc in sim_results:
        print(f"  → {doc.page_content}")

    mmr_results = vectorstore.max_marginal_relevance_search(
        query, k=3, lambda_mult=0.5
    )
    print("\nMMR 检索（兼顾多样性）：")
    for doc in mmr_results:
        print(f"  → {doc.page_content}")


def main():
    print("=" * 60)
    print("📚 Embedding 与 Vector Store 演示")
    print("=" * 60)

    demo_embedding()
    demo_vector_store()
    demo_mmr_search()

    print("\n✅ Embedding 与 Vector Store 演示完成！")


if __name__ == "__main__":
    main()

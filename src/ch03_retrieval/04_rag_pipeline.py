"""
第三章：Retrieval — 完整 RAG 流水线

本脚本演示如何将 Document Loader、Text Splitter、Embedding、Vector Store
和 LLM 组合成一个完整的 RAG（检索增强生成）流水线。

运行方式：
    uv run python src/ch03_retrieval/04_rag_pipeline.py
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def create_rag_chain():
    """创建完整的 RAG Chain"""

    # ========================================
    # 第一步：加载文档
    # ========================================
    from langchain_community.document_loaders import TextLoader

    file_path = os.path.join(DATA_DIR, "sample.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 个文档")

    # ========================================
    # 第二步：切分文档
    # ========================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📦 切分为 {len(chunks)} 个块")

    # ========================================
    # 第三步：创建向量数据库
    # ========================================
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    print(f"💾 向量数据库已创建")

    # 创建检索器（Retriever）
    # as_retriever() 将 VectorStore 包装为 Retriever 接口
    # Retriever 是 LangChain 的标准检索接口，只暴露 invoke() 方法
    # search_type="similarity" 表示使用相似度检索
    # search_kwargs={"k": 2} 表示返回最相关的 2 个文档
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # ========================================
    # 第四步：创建 RAG Prompt
    # ========================================
    # RAG Prompt 的核心思想：
    # 将检索到的文档作为上下文，与用户问题一起送入 LLM
    # 这样 LLM 就能基于这些文档来回答问题，而不是依赖训练数据
    prompt = ChatPromptTemplate.from_template(
        """请根据以下上下文信息来回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""
    )

    # ========================================
    # 第五步：组装 RAG Chain
    # ========================================
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # 定义格式化函数：将检索到的文档列表拼接为字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL 管道：
    # 1. 接收 {"question": "..."} 格式的输入
    # 2. RunnablePassthrough.assign() 保留原始输入，同时添加 context 字段
    #    - question: 原样传递（RunnablePassthrough）
    #    - context: 通过 retriever 检索相关文档，再格式化为字符串
    # 3. prompt: 将 context 和 question 填入模板
    # 4. llm: 调用模型生成回答
    # 5. StrOutputParser: 提取字符串输出
    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever | format_docs)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    print("=" * 60)
    print("📚 完整 RAG 流水线演示")
    print("=" * 60)

    rag_chain = create_rag_chain()

    # ========================================
    # 测试 RAG Chain
    # ========================================
    questions = [
        "LangChain 的核心模块有哪些？",
        "LangChain 的主要应用场景是什么？",
        "今天天气怎么样？",
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")
        answer = rag_chain.invoke({"question": question})
        print(f"💡 回答: {answer}")

    # ========================================
    # RAG 的核心价值
    # ========================================
    # 1. 知识增强：LLM 可以回答训练数据中没有的信息
    # 2. 减少幻觉：基于真实文档回答，而非编造
    # 3. 可追溯：可以知道回答来源于哪个文档
    # 4. 实时性：可以随时更新文档库，无需重新训练模型

    print("\n✅ 完整 RAG 流水线演示完成！")


if __name__ == "__main__":
    main()

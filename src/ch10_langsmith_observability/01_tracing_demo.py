"""
第十章：LangSmith 可观测与评测 — Tracing 示例

运行前可在 .env 中配置：
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=你的 LangSmith API Key
    LANGSMITH_PROJECT=langchain-study

没有 LangSmith Key 时，本脚本仍然可以作为普通 LangChain Chain 运行。

运行方式：
    uv run python src/ch10_langsmith_observability/01_tracing_demo.py
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main() -> None:
    print("=" * 60)
    print("第十章：LangSmith Tracing 示例")
    print("=" * 60)

    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    print(f"LangSmith tracing 是否开启：{tracing_enabled}")
    print(f"LangSmith project：{os.getenv('LANGSMITH_PROJECT', '未设置')}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个 LangChain 教学助手，回答要简洁。"),
            ("human", "用三句话解释什么是 {topic}。"),
        ]
    )
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "LangSmith tracing"})
    print("\n模型回答：")
    print(result)

    print("\n如果配置了 LANGSMITH_TRACING=true，可以在 LangSmith 项目中查看这次调用链路。")


if __name__ == "__main__":
    main()

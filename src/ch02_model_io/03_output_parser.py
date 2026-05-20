"""
第二章：Model I/O — Output Parser 输出解析器

本脚本演示 LangChain 中 Output Parser 的用法，包括：
1. StrOutputParser：字符串输出
2. CommaSeparatedListOutputParser：逗号分隔列表
3. JsonOutputParser：JSON 输出
4. PydanticOutputParser：Pydantic 模型输出

运行方式：
    uv run python src/ch02_model_io/03_output_parser.py
"""

from typing import List

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


def demo_str_output_parser():
    """演示 StrOutputParser — 最简单的解析器"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # StrOutputParser：直接提取 AIMessage 的 content 字段
    # ========================================
    # 没有它：chain.invoke() 返回 AIMessage 对象
    # 有了它：chain.invoke() 返回纯字符串
    # 这是最常用的解析器，几乎每个 Chain 都会用到
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")

    # LCEL 管道：prompt → model → str_parser
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "Python"})
    print(f"StrOutputParser 结果类型: {type(result)}")
    print(f"StrOutputParser 结果: {result}")


def demo_list_output_parser():
    """演示 CommaSeparatedListOutputParser — 列表输出"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # CommaSeparatedListOutputParser
    # ========================================
    # 让模型输出逗号分隔的列表，然后解析为 Python list
    # 它会自动在提示词中添加格式说明（format_instructions）
    parser = CommaSeparatedListOutputParser()

    # 获取格式说明，告诉模型应该怎么输出
    format_instructions = parser.get_format_instructions()
    print(f"格式说明: {format_instructions}")

    prompt = ChatPromptTemplate.from_template(
        "列出5种{category}。\n{format_instructions}"
    )

    chain = prompt | llm | parser

    result = chain.invoke(
        {"category": "编程语言", "format_instructions": format_instructions}
    )
    print(f"\n列表结果类型: {type(result)}")
    print(f"列表结果: {result}")


def demo_json_output_parser():
    """演示 JsonOutputParser — JSON 输出"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # JsonOutputParser
    # ========================================
    # 让模型输出 JSON 格式，然后解析为 Python dict
    # 比 PydanticOutputParser 更轻量，不需要定义模型类
    from langchain_core.output_parsers import JsonOutputParser

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "请用JSON格式回答：列出3个{category}的名称和特点。\n"
        "JSON格式示例: [{{\"name\": \"xxx\", \"feature\": \"xxx\"}}]\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "category": "编程语言",
            "format_instructions": parser.get_format_instructions(),
        }
    )
    print(f"JSON 结果类型: {type(result)}")
    print(f"JSON 结果: {result}")


def demo_pydantic_output_parser():
    """演示 PydanticOutputParser — Pydantic 模型输出（最推荐）"""
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # PydanticOutputParser
    # ========================================
    # 这是功能最强大的解析器，它：
    # 1. 根据 Pydantic 模型自动生成 JSON Schema
    # 2. 将 Schema 嵌入提示词，告诉模型应该输出什么格式
    # 3. 解析模型输出，验证是否符合 Schema
    # 4. 返回类型安全的 Pydantic 模型实例
    from langchain_core.output_parsers import PydanticOutputParser

    # 定义输出结构
    class ProgrammingLanguage(BaseModel):
        """编程语言信息"""

        name: str = Field(description="编程语言名称")
        creator: str = Field(description="创建者")
        year: int = Field(description="发布年份")
        features: List[str] = Field(description="主要特点列表")

    class LanguageList(BaseModel):
        """编程语言列表"""

        languages: List[ProgrammingLanguage] = Field(description="编程语言列表")

    parser = PydanticOutputParser(pydantic_object=LanguageList)

    # 查看自动生成的格式说明
    print(f"Pydantic 格式说明:\n{parser.get_format_instructions()[:300]}...")

    prompt = ChatPromptTemplate.from_template(
        "列出3种主流编程语言的信息。\n{format_instructions}"
    )

    chain = prompt | llm | parser

    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    print(f"\nPydantic 结果类型: {type(result)}")
    print(f"编程语言列表:")
    for lang in result.languages:
        print(f"  - {lang.name} ({lang.year}年, by {lang.creator})")
        print(f"    特点: {', '.join(lang.features)}")


def main():
    print("=" * 60)
    print("📚 Output Parser 演示")
    print("=" * 60)

    demo_str_output_parser()
    print("\n" + "-" * 60)
    demo_list_output_parser()
    print("\n" + "-" * 60)
    demo_json_output_parser()
    print("\n" + "-" * 60)
    demo_pydantic_output_parser()

    print("\n✅ Output Parser 演示完成！")


if __name__ == "__main__":
    main()

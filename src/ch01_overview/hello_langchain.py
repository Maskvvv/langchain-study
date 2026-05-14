"""
第一章：LangChain 概览 — Hello LangChain

本脚本用于验证 LangChain 环境是否正确搭建，并演示最基础的模型调用。

运行方式：
    uv run python src/ch01_overview/hello_langchain.py

前置条件：
    1. 复制 .env.example 为 .env 并填入你的 OPENAI_API_KEY
    2. 确保已安装所有依赖：uv sync
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
# 这一步会读取项目根目录下的 .env 文件，将其中定义的键值对设置为环境变量
# 这样我们就不需要把 API Key 硬编码在代码中，既安全又方便切换
load_dotenv()


def main():
    # ========================================
    # 第一步：创建 Chat Model 实例
    # ========================================
    # ChatOpenAI 是 LangChain 对 OpenAI Chat Completion API 的封装
    # 它实现了 LangChain 的 BaseChatModel 接口，可以与 LangChain 生态无缝集成
    #
    # 关键参数说明：
    #   - model: 指定要使用的模型名称
    #   - temperature: 控制输出的随机性，0=确定性输出，1=高随机性
    #   - api_key: 从环境变量自动读取 OPENAI_API_KEY
    #   - base_url: 从环境变量自动读取 OPENAI_API_BASE（如果设置了的话）
    llm = ChatOpenAI(
        model="kimi-k2.6",
        temperature=1,
    )

    # ========================================
    # 第二步：调用模型
    # ========================================
    # invoke() 是 LangChain 统一的调用接口
    # 输入是一个消息列表，这里我们只传一条人类消息
    # LangChain 会自动将其转换为 OpenAI API 所需的格式
    from langchain_core.messages import HumanMessage

    response = llm.invoke([HumanMessage(content="你好！请用一句话介绍 LangChain")])

    # ========================================
    # 第三步：查看响应
    # ========================================
    # response 是一个 AIMessage 对象，包含以下重要属性：
    #   - content: 模型的文本回复
    #   - response_metadata: 元数据（token 用量、模型名称等）
    #   - id: 消息的唯一标识符
    print("=" * 50)
    print("🤖 模型回复：")
    print(response.content)
    print("=" * 50)
    print("\n📊 响应元数据：")
    print(f"  模型: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"  Token 用量: {response.response_metadata.get('token_usage', {})}")
    print("=" * 50)

    # ========================================
    # 补充：理解 LangChain 的消息类型
    # ========================================
    # LangChain 定义了几种标准消息类型，对应聊天模型的不同角色：
    #   - HumanMessage: 人类用户的消息
    #   - AIMessage: AI 助手的回复
    #   - SystemMessage: 系统指令（设定 AI 的行为）
    #   - ToolMessage: 工具调用的结果
    #
    # 这些消息类型是 LangChain 的核心抽象，贯穿整个框架

    print("\n✅ LangChain 环境搭建成功！你已经完成了第一步！🎉")


if __name__ == "__main__":
    main()

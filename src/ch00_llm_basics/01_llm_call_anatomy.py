"""
第零章：LLM 应用基础 — 一次模型调用的结构

本脚本不依赖真实 API，用普通 Python 数据结构模拟一次 Chat Model 调用。
目的是先理解消息、token、上下文和 temperature 的概念，再进入 LangChain。

运行方式：
    uv run python src/ch00_llm_basics/01_llm_call_anatomy.py
"""

from dataclasses import dataclass


@dataclass
class Message:
    """模拟 LangChain / OpenAI 风格的聊天消息。"""

    role: str
    content: str


def rough_token_count(text: str) -> int:
    """粗略估算 token 数。

    真实 token 计算由模型 tokenizer 决定。这里为了教学，使用一个简单规则：
    - 中文字符大致按 1 个 token 估算
    - 英文按空格分词估算
    这个函数不用于生产，只用于理解“上下文越长，成本越高”。
    """
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
    ascii_words = len([part for part in text.split() if part.strip()])
    return chinese_chars + ascii_words


def build_messages(user_input: str) -> list[Message]:
    """组装一次模型调用要发送的消息列表。"""
    return [
        Message(
            role="system",
            content="你是一个严谨的 AI 工程学习助手，回答要简洁、准确。",
        ),
        Message(role="human", content=user_input),
    ]


def estimate_context_cost(messages: list[Message]) -> None:
    """打印每条消息的粗略 token 数，帮助理解上下文成本。"""
    total = 0
    print("消息列表：")
    for message in messages:
        tokens = rough_token_count(message.content)
        total += tokens
        print(f"  role={message.role:<6} tokens≈{tokens:<3} content={message.content}")

    print(f"\n本次请求输入 token 粗略估算：{total}")
    print("真实项目中应使用模型供应商或 tokenizer 提供的精确 token 统计。")


def explain_temperature() -> None:
    """解释 temperature 对输出稳定性的影响。"""
    print("\ntemperature 选择建议：")
    print("  0.0-0.2：分类、抽取、工具参数生成，追求稳定")
    print("  0.3-0.6：问答、RAG、代码解释，兼顾自然和稳定")
    print("  0.7-1.0：创意写作、头脑风暴，允许更发散")


def main() -> None:
    print("=" * 60)
    print("第零章：LLM 应用基础")
    print("=" * 60)

    user_input = "请用一句话解释 LangChain 是什么。"
    messages = build_messages(user_input)
    estimate_context_cost(messages)
    explain_temperature()

    print("\n核心结论：LangChain 封装的是这些底层概念，而不是替代这些概念。")


if __name__ == "__main__":
    main()

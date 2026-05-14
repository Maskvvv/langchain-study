"""
LangChain 学习项目入口

使用方法：
    uv run python main.py [章节号]

示例：
    uv run python main.py 01   # 运行第一章
    uv run python main.py 02   # 运行第二章
    uv run python main.py      # 显示帮助
"""

import os
import subprocess
import sys


def configure_console_encoding():
    """让 Windows 控制台尽量使用 UTF-8，避免中文和 emoji 输出报错。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


CHAPTERS = {
    "00": ("ch00_llm_basics", "01_llm_call_anatomy.py", "LLM 应用基础"),
    "01": ("ch01_overview", "hello_langchain.py", "LangChain 概览与环境搭建"),
    "02": ("ch02_model_io", "01_chat_model_basics.py", "Model I/O — Chat Model 基础"),
    "03": ("ch03_retrieval", "01_document_loader.py", "Retrieval — Document Loader"),
    "04": ("ch04_chain", "01_lcel_basics.py", "Chain — LCEL 基础"),
    "05": ("ch05_memory", "01_conversation_memory.py", "Memory — 对话记忆"),
    "06": ("ch06_tool_function_calling", "01_tool_definition.py", "Tool & Function Calling"),
    "07": ("ch07_agent", "04_create_agent_v1.py", "Agent — create_agent v1"),
    "08": ("ch08_langgraph", "01_stategraph_basics.py", "LangGraph — StateGraph 基础"),
    "09": ("ch09_production_langgraph", "01_checkpoint_thread.py", "企业级 LangGraph — Checkpoint"),
    "10": ("ch10_langsmith_observability", "01_tracing_demo.py", "LangSmith 可观测与评测"),
    "11": ("ch11_guardrails_security", "01_tool_guardrails.py", "Agent 安全与 Guardrails"),
    "12": ("ch12_enterprise_agent_project", "01_architecture_walkthrough.py", "企业知识库 Agent 项目"),
}


def main():
    configure_console_encoding()

    if len(sys.argv) < 2 or sys.argv[1] not in CHAPTERS:
        print("=" * 60)
        print("📚 LangChain 学习项目")
        print("=" * 60)
        print("\n用法: uv run python main.py [章节号]\n")
        print("可用章节：")
        for ch_num, (folder, script, desc) in CHAPTERS.items():
            print(f"  {ch_num} - {desc}")
            print(f"      uv run python src/{folder}/{script}")
        print()
        return

    ch_num = sys.argv[1]
    folder, script, desc = CHAPTERS[ch_num]

    print(f"📖 运行章节 {ch_num}: {desc}")
    print(f"   文件: src/{folder}/{script}")
    print("=" * 60)
    sys.stdout.flush()

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    subprocess.run(
        [sys.executable, f"src/{folder}/{script}"],
        cwd=sys.path[0] or ".",
        env=env,
    )


if __name__ == "__main__":
    main()

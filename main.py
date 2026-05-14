"""
LangChain 学习项目入口

使用方法：
    uv run python main.py [章节号]

示例：
    uv run python main.py 01   # 运行第一章
    uv run python main.py 02   # 运行第二章
    uv run python main.py      # 显示帮助
"""

import subprocess
import sys


CHAPTERS = {
    "01": ("ch01_overview", "hello_langchain.py", "LangChain 概览与环境搭建"),
    "02": ("ch02_model_io", "01_chat_model_basics.py", "Model I/O — Chat Model 基础"),
    "03": ("ch03_retrieval", "01_document_loader.py", "Retrieval — Document Loader"),
    "04": ("ch04_chain", "01_lcel_basics.py", "Chain — LCEL 基础"),
    "05": ("ch05_memory", "01_conversation_memory.py", "Memory — 对话记忆"),
    "06": ("ch06_tool_function_calling", "01_tool_definition.py", "Tool & Function Calling"),
    "07": ("ch07_agent", "01_react_agent.py", "Agent — ReAct Agent"),
    "08": ("ch08_langgraph", "01_stategraph_basics.py", "LangGraph — StateGraph 基础"),
}


def main():
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

    subprocess.run(
        [sys.executable, f"src/{folder}/{script}"],
        cwd=sys.path[0] or ".",
    )


if __name__ == "__main__":
    main()

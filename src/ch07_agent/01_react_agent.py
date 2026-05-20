"""
第七章：Agent — ReAct Agent

本脚本演示 LangChain 中 ReAct Agent 的构建，包括：
1. 使用 create_tool_calling_agent 创建 Agent
2. 使用 AgentExecutor 运行 Agent
3. ReAct 循环的内部机制

运行方式：
    uv run python src/ch07_agent/01_react_agent.py
"""

from datetime import datetime

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_react_agent():
    """演示 ReAct Agent 的构建与运行"""

    # ========================================
    # 第一步：定义工具
    # ========================================
    # Agent 的能力来源于工具
    # 工具定义越清晰，Agent 的决策越准确

    @tool
    def get_current_time() -> str:
        """获取当前的日期和时间"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        weather_data = {
            "北京": "晴天，25°C，微风，空气质量良好",
            "上海": "多云，28°C，东南风3级，湿度65%",
            "深圳": "阵雨，30°C，南风2级，湿度80%",
            "成都": "阴天，22°C，微风",
        }
        return weather_data.get(city, f"暂无{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果。输入必须是合法的数学表达式"""
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    tools = [get_current_time, search_weather, calculate]

    # ========================================
    # 第二步：创建 Agent Prompt
    # ========================================
    # ReAct Agent 的 Prompt 需要包含：
    # 1. 系统指令（角色设定）
    # 2. 工具信息（自动注入）
    # 3. 用户输入
    # 4. Agent 的中间思考过程（agent_scratchpad）
    #
    # agent_scratchpad 是关键：
    # - 它记录了 Agent 的思考过程和工具调用历史
    # - 每轮循环后，新的 Thought/Action/Observation 都会追加到这里
    # - LLM 通过阅读 scratchpad 来决定下一步行动

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个智能助手，可以使用工具来回答问题。请用中文回答。"),
            ("human", "{input}"),
            # agent_scratchpad 是 AgentExecutor 自动管理的
            # 它存储了 Agent 的中间思考过程
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # ========================================
    # 第三步：创建 Agent
    # ========================================
    # create_tool_calling_agent() 创建一个基于 Function Calling 的 Agent
    # 它会自动将工具定义绑定到 LLM，并处理工具调用的解析
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))
    agent = create_tool_calling_agent(llm, tools, prompt)

    # ========================================
    # 第四步：创建 AgentExecutor
    # ========================================
    # AgentExecutor 是 Agent 的运行器，它管理 ReAct 循环：
    # 1. 调用 Agent 获取下一步行动
    # 2. 执行工具
    # 3. 将结果反馈给 Agent
    # 4. 循环直到 Agent 给出最终回答
    #
    # 关键参数：
    # - max_iterations: 最大循环次数（防止无限循环）
    # - max_execution_time: 最大执行时间
    # - handle_parsing_errors: 解析错误时的处理方式
    # - verbose: 是否打印详细过程
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        handle_parsing_errors=True,
        verbose=True,
    )

    # ========================================
    # 第五步：运行 Agent
    # ========================================
    print("=" * 50)
    print("🤖 ReAct Agent 演示")
    print("=" * 50)

    questions = [
        "北京今天天气怎么样？",
        "现在几点了？上海天气如何？",
        "帮我算一下 (100 + 200) * 3",
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")
        result = agent_executor.invoke({"input": question})
        print(f"💡 回答: {result['output']}")


def demo_agent_with_memory():
    """演示带记忆的 Agent"""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
        }
        return weather_data.get(city, f"暂无{city}的天气信息")

    tools = [search_weather]
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个智能助手，记住用户告诉你的信息。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 添加记忆管理
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "user-1"}}

    # 第一轮
    result1 = agent_with_history.invoke(
        {"input": "我叫小明，我住在北京"}, config=config
    )
    print(f"第一轮: {result1['output']}")

    # 第二轮：Agent 能记住之前的对话
    result2 = agent_with_history.invoke(
        {"input": "我叫什么名字？我住的城市天气怎么样？"}, config=config
    )
    print(f"第二轮: {result2['output']}")


def main():
    print("=" * 60)
    print("📚 ReAct Agent 演示")
    print("=" * 60)

    demo_react_agent()
    print("\n" + "=" * 60)
    demo_agent_with_memory()

    print("\n✅ ReAct Agent 演示完成！")


if __name__ == "__main__":
    main()

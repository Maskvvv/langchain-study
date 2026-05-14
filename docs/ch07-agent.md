# 第七章：Agent — 智能体构建

## 7.1 Agent 是什么？

Agent（智能体）是 LangChain 中最强大的抽象。与 Chain 的固定流程不同，Agent 可以**动态决定**下一步做什么：

```
Chain:  输入 → [固定步骤A] → [固定步骤B] → [固定步骤C] → 输出
Agent:  输入 → [LLM 思考] → [选择行动] → [执行] → [观察结果] → [继续思考...] → 输出
```

核心区别：**Chain 的流程是预定义的，Agent 的流程是 LLM 实时决定的。**

## 7.2 ReAct 模式

ReAct（Reasoning + Acting）是最经典的 Agent 模式：

```
用户: "北京今天天气怎么样？明天适合户外运动吗？"

🤔 思考(Thought): 用户想知道北京天气，我需要调用天气工具
🎬 行动(Action): search_weather(city="北京")
👀 观察(Observation): 北京今天晴天，25°C，微风

🤔 思考: 我已经获取了天气信息，可以回答了
🎬 行动: 返回最终回答
💬 回答: 北京今天晴天，气温25°C，微风。明天如果天气类似，很适合户外运动！
```

### ReAct 的循环

```
┌─────────────────────────────┐
│     Thought（思考）          │
│     "我需要做什么？"         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│     Action（行动）           │
│     "调用某个工具"           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│     Observation（观察）      │
│     "工具返回了什么结果"     │
└──────────┬──────────────────┘
           │
           ▼
    是否需要继续？
      │是 → 回到 Thought
      │否 → 生成最终回答
```

## 7.3 Plan-and-Execute 模式

与 ReAct 的逐步思考不同，Plan-and-Execute 先制定完整计划，再逐步执行：

```
用户: "帮我规划一次北京三日游"

📋 制定计划:
  1. 查询北京未来三天天气
  2. 搜索北京热门景点
  3. 根据天气和景点制定每日行程

▶️ 执行步骤1: 查询天气 → 结果
▶️ 执行步骤2: 搜索景点 → 结果
▶️ 执行步骤3: 制定行程 → 最终回答
```

**ReAct vs Plan-and-Execute 对比**：

| 特性 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| 决策方式 | 每步实时决策 | 先规划再执行 |
| 灵活性 | 高，可随时调整 | 低，按计划执行 |
| 适合场景 | 简单、动态的任务 | 复杂、结构化的任务 |
| 长期规划 | 弱 | 强 |

## 7.4 主流 Agent 框架对比

### LangGraph（LangChain 官方推荐）

- **核心思想**：用状态图（StateGraph）定义 Agent
- **优势**：精细控制、支持循环、人机协作、可视化
- **适合**：复杂工作流、需要精确控制的场景

### CrewAI

- **核心思想**：多角色协作，每个 Agent 有明确的角色和目标
- **优势**：多 Agent 协作简单、角色定义清晰
- **适合**：多角色协作场景（如：研究员+写手+审核员）

### AutoGen (Microsoft)

- **核心思想**：多 Agent 对话，Agent 之间自动协商
- **优势**：人机协作、代码执行、灵活的对话模式
- **适合**：需要人参与决策、代码生成与验证的场景

### 框架选择指南

```
需要精细控制工作流？ → LangGraph
需要多角色协作？     → CrewAI
需要人机对话？       → AutoGen
简单工具调用？       → LangChain AgentExecutor
```

## 7.5 AgentExecutor

AgentExecutor 是 LangChain 早期的 Agent 运行器，封装了 ReAct 循环：

```python
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "..."})
```

它内部实现了：
1. 调用 LLM 获取下一步行动
2. 执行工具
3. 将结果反馈给 LLM
4. 循环直到 LLM 给出最终回答
5. 处理错误和超时

> 💡 **注意**：虽然 AgentExecutor 仍然可用，但 LangChain 官方推荐使用 LangGraph 构建更灵活的 Agent。

## 7.6 LangChain v1 的 create_agent

在 LangChain v1 中，更推荐使用 `create_agent()` 创建 Agent：

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="kimi-k2.6", temperature=0.2)
agent = create_agent(
    model=model,
    tools=[search_weather, calculate],
    system_prompt="你是一个可靠的企业助手。回答前先判断是否需要工具。",
)
```

`create_agent()` 返回的不是早期的 `AgentExecutor`，而是一个基于 LangGraph 的可执行图。它天然支持：

- `checkpointer`：保存会话状态
- `interrupt_before` / `interrupt_after`：在关键节点暂停
- `response_format`：结构化输出
- `middleware`：在模型调用前后注入工程逻辑
- `store`：长期记忆或跨线程数据存储

示例代码：

```bash
uv run python src/ch07_agent/04_create_agent_v1.py
```

## 7.7 AgentExecutor vs create_agent vs LangGraph

| 方案 | 适合场景 | 特点 |
|------|----------|------|
| AgentExecutor | 理解传统 ReAct 和旧项目维护 | 简单，但生产控制力较弱 |
| create_agent | 快速构建现代工具调用 Agent | 底层基于 LangGraph，默认更现代 |
| 手写 LangGraph | 企业级复杂流程 | 最可控，适合审批、恢复、多 Agent、复杂状态 |

学习建议：

1. 用 `AgentExecutor` 理解 ReAct 循环原理。
2. 用 `create_agent` 写简单现代 Agent。
3. 用 LangGraph 承接真正复杂的企业级 Agent。

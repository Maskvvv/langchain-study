# 第八章：LangGraph — 状态图驱动的 Agent

## 8.1 为什么选择 LangGraph？

LangGraph 是 LangChain 官方推荐的 Agent 构建框架，解决了传统 Agent 的几个痛点：

| 痛点 | LangGraph 的解决方案 |
|------|---------------------|
| 流程不可控 | 用状态图精确定义每一步 |
| 无法循环 | 支持条件边和循环 |
| 无法中断 | 支持断点和人机协作 |
| 状态丢失 | 支持持久化检查点 |
| 不可观测 | 支持可视化工作流 |

## 8.2 核心概念

### State（状态）

State 是 Agent 运行过程中维护的数据，通常是一个 TypedDict 或 Pydantic 模型：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表
    next: str                                  # 下一个节点
```

### Node（节点）

节点是执行具体操作的函数，接收 State，返回 State 的更新：

```python
def agent_node(state: State) -> State:
    # 调用 LLM，决定下一步
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Edge（边）

边定义节点之间的转移：

```python
# 普通边：A 执行完一定到 B
graph.add_edge("agent", "tools")

# 条件边：根据状态决定下一个节点
graph.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END,
})
```

## 8.3 构建流程

```
1. 定义 State（状态结构）
2. 定义 Nodes（节点函数）
3. 创建 StateGraph
4. 添加节点和边
5. 编译图
6. 运行图
```

## 8.4 条件路由

条件路由是 LangGraph 最强大的特性之一：

```python
def route_by_intent(state: State) -> str:
    """根据用户意图路由到不同的处理节点"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    elif "需要审核" in last_message.content:
        return "human_review"
    else:
        return "end"
```

## 8.5 人机协作（Human-in-the-Loop）

LangGraph 支持在关键节点暂停执行，等待人类确认：

```
Agent → [生成方案] → ⏸️ 暂停，等待人类确认 → [人类批准] → 继续执行
                                         → [人类拒绝] → 重新生成
```

实现方式：使用 `interrupt_before` 或 `interrupt_after` 参数。

## 8.6 Multi-Agent 编排

LangGraph 可以编排多个 Agent 协作：

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Supervisor│────→│ Researcher│────→│  Writer  │
│  (路由器) │     │ (研究员)  │     │ (写手)   │
└─────────┘     └─────────┘     └─────────┘
     │                                  │
     └──────────── END ←────────────────┘
```

Supervisor 模式：一个中心 Agent 负责路由，将任务分配给专门的子 Agent。

## 8.7 原理深入：StateGraph 的执行机制

1. **初始化**：将输入数据设为初始 State
2. **进入起始节点**：执行节点函数，得到 State 更新
3. **合并 State**：将更新合并到当前 State
4. **路由**：根据边条件决定下一个节点
5. **循环 2-4** 直到到达 END 节点
6. **输出**：返回最终 State

关键设计：
- **不可变 State**：每个节点返回的是 State 的**更新**，而非完整 State
- **Reducer**：定义如何合并更新（如 `add_messages` 是追加而非替换）
- **检查点**：每个节点执行后自动保存 State，支持回滚和恢复

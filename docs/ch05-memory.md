# 第五章：Memory — 对话状态管理

## 5.1 为什么需要 Memory？

LLM 本身是**无状态**的——每次调用都是独立的，不记得之前的对话。但在实际应用中，我们需要多轮对话的上下文连贯性。

Memory 的作用就是在 Chain/Agent 调用之间**持久化对话状态**。

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  第1轮   │    │  第2轮   │    │  第3轮   │
│  用户: Hi │    │ 用户: 记住│    │ 用户: 我 │
│  AI: 你好 │    │ 我叫小明  │    │ 叫什么？ │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
┌─────────────────────────────────────────┐
│              Memory 存储                 │
│  [Human: Hi, AI: 你好,                  │
│   Human: 记住我叫小明, AI: 好的,         │
│   Human: 我叫什么？]                     │
└─────────────────────────────────────────┘
```

## 5.2 Memory 的实现方式

在现代 LangChain 中，Memory 主要有两种实现方式：

### 方式一：手动管理消息历史（推荐）

```python
# 将历史消息通过 MessagesPlaceholder 注入 Prompt
messages = chat_history + [HumanMessage(content=user_input)]
response = llm.invoke(messages)
chat_history.append(HumanMessage(content=user_input))
chat_history.append(response)
```

### 方式二：使用 RunnableWithMessageHistory

```python
# LangChain 提供的自动管理方案
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
)
```

## 5.3 对话记忆类型

| 类型 | 原理 | 适用场景 |
|------|------|----------|
| **完整记忆** | 保存所有历史消息 | 短对话、调试 |
| **滑动窗口** | 只保留最近 K 轮对话 | 一般对话 |
| **摘要记忆** | 用 LLM 对历史对话生成摘要 | 长对话 |
| **实体记忆** | 提取对话中的关键实体 | 需要记住特定信息 |

## 5.4 原理深入：为什么需要限制记忆长度？

1. **上下文窗口限制**：每个模型有最大 token 限制（如 GPT-4o 为 128K）
2. **成本**：每轮对话都要发送完整历史，token 越多费用越高
3. **注意力稀释**：太多历史信息会干扰模型对当前问题的关注

因此，滑动窗口和摘要记忆是更实用的方案。

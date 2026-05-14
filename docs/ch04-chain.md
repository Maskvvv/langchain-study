# 第四章：Chain — 链式调用与编排

## 4.1 Chain 是什么？

Chain 是 LangChain 的核心编排机制，将多个组件串联成一个工作流：

```
输入 → [组件A] → [组件B] → [组件C] → 输出
```

在 LangChain 早期版本中，Chain 是具体的类（如 LLMChain、SequentialChain）。在现代版本中，**LCEL（LangChain Expression Language）** 取代了这些类，成为构建 Chain 的推荐方式。

## 4.2 LCEL（LangChain Expression Language）

LCEL 是 LangChain 的声明式语法，使用管道符 `|` 串联组件：

```python
chain = prompt | model | output_parser
```

### 核心优势

| 特性 | 说明 |
|------|------|
| **简洁** | 一行代码定义完整工作流 |
| **统一接口** | 所有组件都实现 Runnable 协议 |
| **流式支持** | 自动支持 stream()、astream() |
| **并行执行** | RunnableParallel 支持并行 |
| **类型安全** | 输入输出类型可推断 |

## 4.3 Runnable 协议

Runnable 是 LCEL 的基础，所有组件都实现了这个协议：

```
┌─────────────────────────────────┐
│         Runnable 协议            │
├─────────────────────────────────┤
│ invoke()      同步单次调用       │
│ stream()      同步流式调用       │
│ batch()       同步批量调用       │
│ ainvoke()     异步单次调用       │
│ astream()     异步流式调用       │
│ abatch()      异步批量调用       │
├─────────────────────────────────┤
│ | (pipe)      串联组合           │
│ assign()      添加新字段         │
│ map()         对列表元素逐一执行  │
└─────────────────────────────────┘
```

## 4.4 组合模式

### 串联（Sequential）

```python
chain = prompt | model | parser
```

### 并行（Parallel）

```python
parallel_chain = RunnableParallel({
    "joke": joke_chain,
    "poem": poem_chain,
})
```

### 分支（Branch）

```python
branch_chain = chain.with_fallbacks([backup_model])
```

## 4.5 原理深入：管道符 `|` 的实现

Python 的 `|` 运算符对应 `__or__()` 魔术方法。LangChain 在 Runnable 中重载了这个方法：

```python
def __or__(self, other):
    return RunnableSequence(self, other)
```

所以 `prompt | model | parser` 等价于：

```python
RunnableSequence(RunnableSequence(prompt, model), parser)
```

当调用 `chain.invoke(input)` 时，数据会依次流过每个组件。

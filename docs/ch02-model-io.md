# 第二章：Model I/O — 与大模型对话

## 2.1 Model I/O 概述

Model I/O 是 LangChain 最基础的模块，负责与大语言模型的所有交互。它由三个核心部分组成：

```
用户输入 → [Prompt Template] → 格式化提示词 → [Chat Model / LLM] → 原始输出 → [Output Parser] → 结构化数据
```

| 组件 | 作用 | 类比 |
|------|------|------|
| **Prompt** | 管理和格式化输入提示词 | 给 AI 写的信 |
| **Model** | 调用大语言模型生成回复 | AI 大脑 |
| **Output Parser** | 解析模型输出为结构化数据 | 翻译官 |

## 2.2 Chat Model vs LLM

LangChain 区分两种模型接口：

- **Chat Model**：以聊天消息列表为输入，返回 AI 消息。这是当前主流方式。
  - 例：`ChatOpenAI`、`ChatAnthropic`、`ChatGoogleGenerativeAI`
- **LLM**：以纯文本为输入，返回纯文本。这是旧版接口。
  - 例：`OpenAI`（text-davinci-003）

> 💡 **建议**：优先使用 Chat Model，它是现代 LLM 的标准接口。

## 2.3 Prompt Template

Prompt Template（提示词模板）是将变量注入提示词的机制：

```
模板: "请将以下{source_language}文本翻译为{target_language}：\n{text}"
变量: source_language="中文", target_language="英文", text="你好世界"
结果: "请将以下中文文本翻译为英文：\n你好世界"
```

### 类型

1. **ChatPromptTemplate**：聊天提示词模板，支持多种消息类型
2. **PromptTemplate**：纯文本提示词模板（旧版）
3. **MessagesPlaceholder**：消息占位符，用于插入历史消息

## 2.4 Output Parser

Output Parser 将模型的自由文本输出转换为结构化数据：

| 解析器 | 用途 |
|--------|------|
| **StrOutputParser** | 直接获取字符串输出（最简单） |
| **CommaSeparatedListOutputParser** | 解析逗号分隔的列表 |
| **JsonOutputParser** | 解析 JSON 格式输出 |
| **PydanticOutputParser** | 解析为 Pydantic 模型（最常用） |

## 2.5 流式输出

流式输出（Streaming）是提升用户体验的关键技术：

- **原理**：模型逐 token 生成，服务端每生成一个 token 就立即发送给客户端
- **好处**：用户无需等待完整响应，可以实时看到生成过程
- **LangChain 实现**：通过 `.stream()` 方法和回调机制实现

## 2.6 核心原理：LCEL（LangChain Expression Language）

LCEL 是 LangChain 的核心语法，使用管道符 `|` 将组件串联：

```python
chain = prompt | model | output_parser
result = chain.invoke({"topic": "AI"})
```

这背后的原理是 **Runnable 协议**——所有组件都实现了 `invoke()`、`stream()`、`batch()` 等统一接口，因此可以像管道一样自由组合。

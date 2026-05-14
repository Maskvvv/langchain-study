# LangChain Agent 开发学习路线

这个项目面向想系统学习 LangChain 并进入 Agent 开发方向的新手。路线不是只讲 API 怎么调，而是按“原理 -> 代码 -> 工程化 -> 企业级 Agent”的节奏推进。

## 学习目标

完成本项目后，你应该能够：

- 理解 LLM 应用的基本概念：token、上下文窗口、temperature、消息角色、结构化输出。
- 掌握 LangChain 的核心抽象：Chat Model、Prompt、Output Parser、Runnable、Tool、Retriever。
- 能用 LCEL 组合基础 Chain，能构建简单 RAG 应用。
- 理解 Agent 的 ReAct、Tool Calling、Plan-and-Execute、Multi-Agent 等常见模式。
- 能用 LangGraph 构建可控、可恢复、支持人工审核的 Agent 工作流。
- 知道企业级 Agent 需要补齐的工程能力：可观测、评测、安全、权限、部署、测试。

## 环境准备

```bash
uv sync
copy .env.example .env
```

然后在 `.env` 中配置你的模型服务：

```env
OPENAI_API_KEY=你的 API Key
OPENAI_API_BASE=https://api.openai.com/v1
```

如果你使用兼容 OpenAI API 的服务，例如 Kimi、DeepSeek、通义千问等，通常只需要修改 `OPENAI_API_BASE` 和代码中的 `model` 名称。

## 运行方式

查看所有章节：

```bash
uv run python main.py
```

运行指定章节：

```bash
uv run python main.py 01
uv run python main.py 08
uv run python main.py 09
```

也可以直接运行某个示例文件：

```bash
uv run python src/ch08_langgraph/01_stategraph_basics.py
```

## 章节路线

| 阶段 | 章节 | 主题 | 目标 |
|------|------|------|------|
| 基础认知 | 00 | LLM 应用基础 | 先理解 token、消息、上下文、成本和模型参数 |
| 入门 | 01 | LangChain 概览 | 搭环境，跑通第一个模型调用 |
| 核心接口 | 02 | Model I/O | 掌握 Chat Model、Prompt、Parser、Streaming |
| 知识库 | 03 | Retrieval / RAG | 理解文档加载、切分、Embedding、Vector Store |
| 组合能力 | 04 | LCEL Chain | 用 Runnable 协议把组件串起来 |
| 状态管理 | 05 | Memory | 管理多轮对话和长期状态 |
| 工具调用 | 06 | Tool & Function Calling | 让模型调用外部函数和系统能力 |
| Agent 模式 | 07 | Agent | 理解 ReAct、计划执行、框架选型 |
| 图编排 | 08 | LangGraph | 用 StateGraph 构建可控 Agent |
| 生产化 | 09 | 企业级 LangGraph | 持久化、恢复、人工审核、错误处理 |
| 可观测 | 10 | LangSmith | tracing、dataset、evaluation、回归测试 |
| 安全 | 11 | Guardrails | 工具权限、输入输出校验、敏感操作审批 |
| 项目实战 | 12 | 企业知识库 Agent | 把 RAG、工具、图编排、评测和部署串成完整项目 |

## 推荐学习方法

1. 先读 `docs/` 下对应章节的 Markdown，理解概念和原理。
2. 再运行 `src/` 下对应示例，观察输入输出。
3. 修改参数，例如 `temperature`、`chunk_size`、检索 `k` 值、Agent 最大循环次数。
4. 每学完一个阶段，自己写一个小练习，不要只看不改。
5. 最后用第 12 章做一个完整项目，作为求职作品集的一部分。

## 企业级 Agent 开发重点

企业里的 Agent 通常不是一个“聊天脚本”，而是一套可运营的软件系统。重点包括：

- **可控流程**：复杂任务优先用 LangGraph，而不是把所有决策都交给模型自由发挥。
- **可观测性**：所有模型调用、工具调用、错误、耗时、token 成本都要能追踪。
- **可评测性**：用固定测试集评估回答质量、工具选择准确率、幻觉率和回归风险。
- **安全边界**：敏感工具必须有权限、审批、参数校验和审计日志。
- **状态持久化**：长任务、多轮对话、人机协作都需要 thread/checkpoint 机制。
- **部署集成**：Agent 要能作为 API、后台任务或企业系统插件运行。

## 当前项目状态

本仓库已经包含 LangChain 核心学习路径和 LangGraph 入门示例。后续优化重点是把每一章补成“概念讲解 + 可运行代码 + 练习题 + 企业实践提醒”的完整课程单元。

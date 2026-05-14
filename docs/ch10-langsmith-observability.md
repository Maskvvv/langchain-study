# 第十章：LangSmith 可观测与评测

Agent 开发不能只看“这次回答还不错”。企业项目需要知道：为什么这么回答、调用了哪些工具、花了多少钱、是否比上一个版本更好。

LangSmith 主要解决四类问题：

1. **Tracing**：追踪每次模型调用、工具调用、Chain 和 Graph 节点。
2. **Debugging**：定位 prompt、检索、工具参数、输出解析的问题。
3. **Evaluation**：用数据集和评分器评估质量。
4. **Monitoring**：上线后持续观察错误率、延迟、成本和用户反馈。

## 10.1 Tracing

开启 tracing 后，你可以看到一次请求的完整链路：

```
用户问题
  ↓
Prompt Template
  ↓
Retriever
  ↓
LLM
  ↓
Output Parser
  ↓
最终回答
```

对于 Agent，还能看到每一次工具选择、工具参数和工具返回结果。

## 10.2 环境变量

通常需要配置：

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的 LangSmith Key
LANGSMITH_PROJECT=langchain-study
```

示例代码见：

```bash
uv run python src/ch10_langsmith_observability/01_tracing_demo.py
```

## 10.3 Evaluation

评测的目标不是证明模型“聪明”，而是回答这些工程问题：

- 修改 prompt 后，准确率有没有下降？
- 换模型后，成本和质量怎么变化？
- RAG 检索是否真的找到了正确材料？
- Agent 是否调用了正确工具？
- 是否产生了幻觉？

## 10.4 常见评测指标

| 指标 | 说明 |
|------|------|
| 答案正确性 | 回答是否符合参考答案 |
| 忠实度 | 回答是否基于检索上下文 |
| 工具选择准确率 | 是否选对工具 |
| 参数准确率 | 工具参数是否正确 |
| 幻觉率 | 是否编造不存在的信息 |
| 延迟 | 响应耗时 |
| 成本 | token 和工具调用成本 |

## 10.5 企业实践建议

- 每个核心 Agent 都要有一组固定评测集。
- 每次改 prompt、换模型、改检索参数，都跑一遍回归评测。
- 不要只看平均分，要看失败样例。
- 对高风险流程，评测集中要包含边界输入和恶意输入。
- 线上 trace 要能关联用户、会话、版本号和模型配置。

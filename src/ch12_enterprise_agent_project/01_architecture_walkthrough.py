"""
第十二章：企业知识库 Agent 项目 — 架构走读

这个脚本打印企业级 Agent 项目的模块拆分和建设顺序。

运行方式：
    uv run python src/ch12_enterprise_agent_project/01_architecture_walkthrough.py
"""


def main() -> None:
    print("=" * 60)
    print("第十二章：企业知识库 Agent 项目")
    print("=" * 60)

    modules = [
        ("接口层", "FastAPI / Web UI，负责用户请求、鉴权、会话 ID"),
        ("编排层", "LangGraph StateGraph，负责流程、路由、暂停与恢复"),
        ("模型层", "ChatOpenAI / 兼容 OpenAI 的模型服务"),
        ("知识层", "Document Loader、Text Splitter、Embedding、Vector Store"),
        ("工具层", "订单查询、消息发送、任务创建等业务工具"),
        ("安全层", "权限校验、参数校验、敏感操作审批、审计日志"),
        ("观测层", "LangSmith tracing、评测集、成本和延迟统计"),
    ]

    print("推荐模块拆分：")
    for index, (name, description) in enumerate(modules, start=1):
        print(f"  {index}. {name}: {description}")

    milestones = [
        "先做只读 RAG，不接写操作工具",
        "加入低风险查询工具，并记录工具调用日志",
        "用 LangGraph 显式编排意图识别、检索、工具、回答节点",
        "为高风险工具加入人工审批节点",
        "接入 LangSmith tracing，建立第一版评测集",
        "用 FastAPI 暴露接口，补充权限和审计",
    ]

    print("\n建议开发顺序：")
    for index, milestone in enumerate(milestones, start=1):
        print(f"  M{index}: {milestone}")

    print("\n核心原则：先做可靠的窄场景，再逐步扩大 Agent 能力边界。")


if __name__ == "__main__":
    main()

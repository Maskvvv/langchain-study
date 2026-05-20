"""
第七章：Agent — Plan-and-Execute 模式

本脚本演示 LangChain 中 Plan-and-Execute Agent 的构建，包括：
1. 手动实现 Plan-and-Execute 模式
2. 与 ReAct 模式的对比

运行方式：
    uv run python src/ch07_agent/02_plan_and_execute.py
"""

from typing import List

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


def demo_plan_and_execute():
    """演示 Plan-and-Execute 模式"""

    # ========================================
    # Plan-and-Execute 的原理
    # ========================================
    # 与 ReAct 的逐步思考不同，Plan-and-Execute 分为两个阶段：
    #
    # 阶段一：Planning（规划）
    #   LLM 根据任务生成一个步骤列表
    #
    # 阶段二：Execution（执行）
    #   按顺序执行每个步骤，可以调用工具
    #   执行结果可以反馈给 Planner，调整后续计划
    #
    # 优势：
    # - 更适合复杂、多步骤的任务
    # - 有全局视野，不会迷失方向
    # - 可以在执行过程中调整计划

    # 定义工具
    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的天气"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "深圳": "阵雨，30°C",
        }
        return weather_data.get(city, f"暂无{city}的天气信息")

    @tool
    def search_attractions(city: str) -> str:
        """搜索城市的热门景点"""
        attractions = {
            "北京": "故宫、长城、颐和园、天坛、鸟巢",
            "上海": "外滩、东方明珠、豫园、迪士尼、南京路",
            "深圳": "世界之窗、欢乐谷、大梅沙、华强北",
        }
        return attractions.get(city, f"暂无{city}的景点信息")

    tools = [search_weather, search_attractions]
    tools_map = {t.name: t for t in tools}

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=float(os.getenv("LLM_TEMPERATURE")))

    # ========================================
    # 阶段一：Planning（规划）
    # ========================================

    class PlanStep(BaseModel):
        """计划中的一个步骤"""
        step: int = Field(description="步骤编号")
        action: str = Field(description="要执行的操作描述")
        tool: str | None = Field(default=None, description="需要使用的工具名称（如果有）")
        tool_args: dict | None = Field(default=None, description="工具参数")

    class Plan(BaseModel):
        """执行计划"""
        goal: str = Field(description="任务目标")
        steps: List[PlanStep] = Field(description="步骤列表")

    planner = llm.with_structured_output(Plan)

    task = "帮我规划一次北京一日游，要考虑天气情况"
    print(f"📋 任务: {task}\n")

    # 生成计划
    plan_prompt = ChatPromptTemplate.from_template(
        "请为以下任务制定执行计划。可用的工具：{tools}\n\n任务：{task}"
    )

    tools_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    plan = (plan_prompt | planner).invoke({"task": task, "tools": tools_desc})

    print(f"🎯 目标: {plan.goal}")
    print(f"📝 计划:")
    for step in plan.steps:
        tool_info = f" [工具: {step.tool}({step.tool_args})]" if step.tool else ""
        print(f"  {step.step}. {step.action}{tool_info}")

    # ========================================
    # 阶段二：Execution（执行）
    # ========================================
    print(f"\n▶️ 开始执行计划:")

    results = {}
    for step in plan.steps:
        print(f"\n  步骤 {step.step}: {step.action}")

        if step.tool and step.tool in tools_map:
            # 执行工具
            tool_result = tools_map[step.tool].invoke(step.tool_args or {})
            print(f"    🔧 工具结果: {tool_result}")
            results[step.step] = tool_result
        else:
            # 不需要工具，由 LLM 生成
            print(f"    🤔 由 LLM 处理...")
            results[step.step] = "已处理"

    # ========================================
    # 阶段三：生成最终回答
    # ========================================
    results_text = "\n".join(f"步骤{ k}: {v}" for k, v in results.items())
    final_prompt = ChatPromptTemplate.from_template(
        "任务: {task}\n\n执行结果:\n{results}\n\n请根据以上结果，生成最终的完整回答："
    )
    final_chain = final_prompt | llm
    final_answer = final_chain.invoke({"task": task, "results": results_text})

    print(f"\n💡 最终回答:\n{final_answer.content}")


def demo_react_vs_plan():
    """对比 ReAct 和 Plan-and-Execute"""
    print("\n" + "=" * 50)
    print("📊 ReAct vs Plan-and-Execute 对比")
    print("=" * 50)

    print("""
┌────────────────┬──────────────────┬──────────────────┐
│     特性       │     ReAct        │ Plan-and-Execute │
├────────────────┼──────────────────┼──────────────────┤
│ 决策方式       │ 每步实时决策     │ 先规划再执行     │
│ 全局视野       │ 弱               │ 强               │
│ 灵活性         │ 高               │ 中               │
│ 适合任务       │ 简单、动态       │ 复杂、结构化     │
│ Token 消耗     │ 较少             │ 较多（需规划）   │
│ 长期规划       │ 弱               │ 强               │
│ 错误恢复       │ 自然适应         │ 需要重新规划     │
└────────────────┴──────────────────┴──────────────────┘

💡 选择建议：
  - 简单问答/单步操作 → ReAct
  - 复杂多步骤任务 → Plan-and-Execute
  - 需要人参与决策 → LangGraph（下一章）
""")


def main():
    print("=" * 60)
    print("📚 Plan-and-Execute 演示")
    print("=" * 60)

    demo_plan_and_execute()
    demo_react_vs_plan()

    print("\n✅ Plan-and-Execute 演示完成！")


if __name__ == "__main__":
    main()

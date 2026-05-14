"""
第六章：Tool & Function Calling — 工具定义

本脚本演示 LangChain 中 Tool 的定义方式，包括：
1. @tool 装饰器定义工具
2. StructuredTool 类定义工具
3. 工具的元数据与描述

运行方式：
    uv run python src/ch06_tool_function_calling/01_tool_definition.py
"""

import json
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

load_dotenv()


def demo_tool_decorator():
    """演示 @tool 装饰器 — 最常用的工具定义方式"""
    # ========================================
    # @tool 装饰器
    # ========================================
    # 将普通 Python 函数转换为 LangChain Tool
    # LangChain 会自动：
    # 1. 使用函数名作为工具名称
    # 2. 使用函数的 docstring 作为工具描述
    # 3. 使用函数的类型注解生成参数 Schema
    #
    # ⚠️ docstring 非常重要！LLM 根据描述决定何时调用工具

    @tool
    def search_weather(city: str) -> str:
        """查询指定城市的当前天气信息"""
        # 在实际应用中，这里会调用天气 API
        # 现在用模拟数据演示
        weather_data = {
            "北京": "晴天，25°C，微风",
            "上海": "多云，28°C，东南风3级",
            "深圳": "阵雨，30°C，南风2级",
        }
        return weather_data.get(city, f"未找到{city}的天气信息")

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式的结果。输入必须是合法的数学表达式，如 '2+3*4'"""
        try:
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # 查看工具的属性
    print(f"🔧 工具名称: {search_weather.name}")
    print(f"📝 工具描述: {search_weather.description}")
    print(f"📋 参数 Schema: {json.dumps(search_weather.args_schema.schema(), indent=2, ensure_ascii=False)}")

    # 直接调用工具（测试用）
    print(f"\n调用结果: {search_weather.invoke({'city': '北京'})}")
    print(f"调用结果: {calculate.invoke({'expression': '2+3*4'})}")


def demo_tool_with_pydantic():
    """演示使用 Pydantic 模型定义工具参数"""
    # ========================================
    # 使用 Pydantic 模型定义复杂参数
    # ========================================
    # 当工具参数较复杂时，可以用 Pydantic 模型来定义
    # 这比纯类型注解更清晰、更强大

    class SearchInput(BaseModel):
        """搜索工具的输入参数"""
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=5, description="最大返回结果数")
        language: Optional[str] = Field(default=None, description="结果语言过滤，如 'zh' 或 'en'")

    @tool(args_schema=SearchInput)
    def search_web(query: str, max_results: int = 5, language: Optional[str] = None) -> str:
        """在网络上搜索信息"""
        # 模拟搜索结果
        results = [
            f"结果{i + 1}: 关于'{query}'的搜索结果"
            for i in range(max_results)
        ]
        lang_info = f" (语言: {language})" if language else ""
        return f"找到 {max_results} 条结果{lang_info}:\n" + "\n".join(results)

    print(f"\n🔧 工具名称: {search_web.name}")
    print(f"📋 参数 Schema: {json.dumps(search_web.args_schema.schema(), indent=2, ensure_ascii=False)}")
    print(f"\n调用结果: {search_web.invoke({'query': 'LangChain', 'max_results': 3, 'language': 'zh'})}")


def demo_structured_tool():
    """演示 StructuredTool — 更灵活的工具定义"""
    # ========================================
    # StructuredTool
    # ========================================
    # 当需要更灵活地定义工具时使用
    # 例如：工具函数已经存在，不想用装饰器修改

    def send_email(to: str, subject: str, body: str) -> str:
        # 模拟发送邮件
        return f"邮件已发送给 {to}\n主题: {subject}\n内容: {body[:50]}..."

    class EmailInput(BaseModel):
        """邮件输入参数"""
        to: str = Field(description="收件人邮箱地址")
        subject: str = Field(description="邮件主题")
        body: str = Field(description="邮件正文")

    email_tool = StructuredTool.from_function(
        func=send_email,
        name="send_email",
        description="发送电子邮件给指定收件人",
        args_schema=EmailInput,
    )

    print(f"\n🔧 工具名称: {email_tool.name}")
    print(f"📝 工具描述: {email_tool.description}")
    print(f"\n调用结果: {email_tool.invoke({'to': 'test@example.com', 'subject': '你好', 'body': '这是一封测试邮件'})}")


def demo_tool_best_practices():
    """演示工具定义的最佳实践"""
    # ========================================
    # 工具定义的最佳实践
    # ========================================

    # ✅ 好的描述：清晰、具体、包含使用场景
    @tool
    def search_restaurant(cuisine: str, location: str) -> str:
        """根据菜系和位置搜索餐厅。
        cuisine: 菜系类型，如 '川菜'、'日料'、'西餐'
        location: 位置，如 '北京朝阳区'
        返回附近评分最高的3家餐厅。"""
        return f"在{location}找到3家{cuisine}餐厅"

    # ❌ 差的描述：模糊、不具体
    # @tool
    # def search(c: str, l: str) -> str:
    #     """搜索东西"""
    #     return "结果"

    # ✅ 好的参数名：语义清晰
    # ✅ 好的参数描述：说明格式和取值范围

    print("\n💡 工具定义最佳实践：")
    print("  1. 描述要清晰具体，包含使用场景")
    print("  2. 参数名要有语义，不要用缩写")
    print("  3. 使用 Field(description=...) 说明参数格式")
    print("  4. 返回值要信息丰富，方便 LLM 理解")
    print("  5. 工具粒度要适中，不要太大也不要太小")


def main():
    print("=" * 60)
    print("📚 Tool 定义演示")
    print("=" * 60)

    demo_tool_decorator()
    demo_tool_with_pydantic()
    demo_structured_tool()
    demo_tool_best_practices()

    print("\n✅ Tool 定义演示完成！")


if __name__ == "__main__":
    main()

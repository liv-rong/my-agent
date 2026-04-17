"""
Step 5（扩展）：LangGraph ReAct Agent + 交互循环

教程里最简单的单次 LLM 调用见 GUIDE「Step 5」；本文件在同环境上接入工具与对话。
运行方式：python -m my_agent.step5_basic

依赖 .env：OPENAI_API_KEY、OPENAI_BASE_URL（与教程一致）
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from my_agent.tools.search import calculate, search_weather

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="claude-opus-4-6",
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

tools = [search_weather, calculate]

# 3. 创建记忆存储（内存版，重启后丢失）
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个乐于助人的AI助手。请用中文回复用户的问题。如果需要查询天气或计算，请使用提供的工具。",
    checkpointer=memory,  # 关键：传入记忆组件
)

config = {"configurable": {"thread_id": "user-001"}}
print("=" * 50)
print("🤖 带记忆的 Agent 已启动！输入 'quit' 退出")
print("=" * 50)

while True:
    user_input = input("\n你：")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("再见！👋")
        break

    # 调用 Agent（传入 config 以启用记忆）
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,  # 关键：传入 config
    )

    # print(result)
    # print(result.tool_calls)

    final_message = result["messages"][-1]
    print(f"\n🤖 Agent：{final_message.content}")

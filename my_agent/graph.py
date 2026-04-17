"""
LangGraph 服务入口
用于 langgraph dev 命令启动
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from my_agent.tools.search import search_weather, calculate

load_dotenv()

llm = ChatOpenAI(
    model="claude-opus-4-6",
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
tools = [search_weather, calculate]

# 导出 graph 变量，供 LangGraph CLI 使用
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个乐于助人的AI助手。请用中文回复。",
)

"""
Step 5：最简单的 LLM 调用
运行方式：python -m my_agent.step5_basic
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载 .env 中的环境变量
load_dotenv()

# 2. 创建 LLM 实例
#    model 可以换成 "gpt-3.5-turbo"、"gpt-4o" 等
llm = ChatOpenAI(
    model="claude-opus-4-6",
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
)


response = llm.invoke("你好！请用一句话介绍什么是 AI Agent。")


print("=" * 50)
print("LLM 回复：")
print(response.content)
print("=" * 50)

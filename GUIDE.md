# 从零搭建 AI Agent 智能体 — 手把手教程

> 跟着下面的步骤，一步一步操作，最终你会拥有一个能调用工具、自主决策的 AI Agent。

---

## 目录

| 步骤 | 内容 | 预计时间 |
|------|------|----------|
| Step 1 | 环境准备 | 5 分钟 |
| Step 2 | 安装依赖 | 3 分钟 |
| Step 3 | 配置 API Key | 2 分钟 |
| Step 4 | 创建项目结构 | 2 分钟 |
| Step 5 | 写一个最简单的 LLM 调用 | 10 分钟 |
| Step 6 | 给 Agent 添加工具（Tool） | 15 分钟 |
| Step 7 | 用 LangGraph 构建完整 Agent | 20 分钟 |
| Step 8 | 添加对话记忆 | 10 分钟 |
| Step 9 | 用 LangGraph CLI 启动服务 | 5 分钟 |
| Step 10 | 下一步学习方向 | — |

---

## Step 1：环境准备

### 1.1 确认 Python 版本

```bash
python --version
# 需要 Python >= 3.11
```

### 1.2 创建虚拟环境（你已经有了，跳过即可）

```bash
# 如果还没有 venv，执行：
python -m venv venv

# 激活虚拟环境
# macOS / Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

> ✅ 检查点：终端提示符前面出现 `(venv)` 就说明激活成功了。

---

## Step 2：安装依赖

逐个安装，这样你能清楚每个包的作用：

```bash
# 核心框架：LangChain（LLM 编排框架）
pip install langchain

# LangChain 的 OpenAI 集成（用于调用 ChatGPT / 兼容接口）
pip install langchain-openai

# LangGraph（Agent 状态图框架，LangChain 团队出品）
pip install langgraph

# LangGraph CLI（用于本地启动 Agent 服务）
pip install "langgraph-cli[inmem]"

# 如果你想用其他模型，可以按需安装：
# pip install langchain-anthropic   # Claude
# pip install langchain-google-genai # Gemini

# 环境变量管理
pip install python-dotenv
```

安装完成后，冻结依赖：

```bash
pip freeze > requirements.txt
```

> ✅ 检查点：运行 `pip list | grep langgraph` 能看到 langgraph 已安装。

---

## Step 3：配置 API Key

在项目根目录创建 `.env` 文件：

```bash
touch .env
```

编辑 `.env`，填入你的 Key：

```env
# === 必填：第三方中转站配置 ===
OPENAI_API_KEY=sk-你的中转站Key
OPENAI_BASE_URL=https://你的中转站地址/v1

# === 可选：LangSmith 追踪调试（强烈推荐开启）===
# 注册地址：https://smith.langchain.com
LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-agent
```

**安全提醒**：创建 `.gitignore` 避免泄露 Key：

```bash
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

> ✅ 检查点：`.env` 文件存在，且已被 `.gitignore` 排除。

---

## Step 4：创建项目结构

```bash
mkdir -p my_agent/tools
touch my_agent/__init__.py
touch my_agent/tools/__init__.py
```

最终目录结构：

```
my-agent/
├── .env                  # API Key 配置
├── .gitignore
├── requirements.txt
├── langgraph.json        # LangGraph CLI 配置（Step 9 创建）
├── my_agent/
│   ├── __init__.py
│   ├── step5_basic.py    # Step 5：最简单的 LLM 调用
│   ├── step6_tools.py    # Step 6：带工具的 Agent
│   ├── step7_graph.py    # Step 7：LangGraph 完整 Agent
│   ├── step8_memory.py   # Step 8：带记忆的 Agent
│   └── tools/
│       ├── __init__.py
│       └── search.py     # 自定义工具
└── venv/
```

> ✅ 检查点：`ls my_agent/` 能看到 `__init__.py` 和 `tools/` 目录。

---

## Step 5：写一个最简单的 LLM 调用

**目标**：确认 API Key 能用，LLM 能正常回复。

创建文件 `my_agent/step5_basic.py`：

```python
"""
Step 5：最简单的 LLM 调用
运行方式：python -m my_agent.step5_basic
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载 .env 中的环境变量
load_dotenv()

# 2. 创建 LLM 实例
#    通过第三方中转站调用 Claude Opus
import os
llm = ChatOpenAI(model="claude-opus-4-6", temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))

# 3. 发送一条消息
response = llm.invoke("你好！请用一句话介绍什么是 AI Agent。")

# 4. 打印结果
print("=" * 50)
print("LLM 回复：")
print(response.content)
print("=" * 50)
```

运行测试：

```bash
python -m my_agent.step5_basic
```

> ✅ 检查点：能看到 LLM 的中文回复，没有报错。
>
> ❌ 常见问题：
> - `AuthenticationError` → 检查 `.env` 中的 API Key 是否正确
> - `Connection Error` → 如果在国内，可能需要配置代理地址

---

## Step 6：给 Agent 添加工具（Tool）

**目标**：让 Agent 不只是聊天，还能**调用工具**获取外部信息。

### 6.1 先理解什么是 Tool

```
用户提问 → Agent 思考 → 决定调用哪个工具 → 拿到工具结果 → 组织回答
```

Agent 和普通聊天机器人的最大区别就是：**Agent 能自主决定是否调用工具、调用哪个工具。**

### 6.2 创建自定义工具

创建文件 `my_agent/tools/search.py`：

```python
"""
自定义工具示例
"""

from langchain_core.tools import tool


@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气信息。当用户询问天气时使用此工具。"""
    # 这里用模拟数据，实际项目中你可以调用真实天气 API
    weather_data = {
        "北京": "晴天，25°C，微风",
        "上海": "多云，22°C，东南风3级",
        "深圳": "阵雨，28°C，湿度85%",
    }
    return weather_data.get(city, f"抱歉，暂无 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。当用户需要数学计算时使用此工具。"""
    try:
        result = eval(expression)  # 注意：生产环境不要用 eval
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"
```

### 6.3 把工具绑定到 LLM

创建文件 `my_agent/step6_tools.py`：

```python
"""
Step 6：带工具调用的 Agent
运行方式：python -m my_agent.step6_tools
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from my_agent.tools.search import search_weather, calculate

load_dotenv()

# 1. 创建 LLM
llm = ChatOpenAI(model="claude-opus-4-6", temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))

# 2. 把工具绑定到 LLM（关键步骤！）
tools = [search_weather, calculate]
llm_with_tools = llm.bind_tools(tools)

# 3. 测试：LLM 会自动判断是否需要调用工具
response = llm_with_tools.invoke("北京今天天气怎么样？")

print("=" * 50)
print("LLM 回复类型：", type(response))
print()

# 4. 查看 LLM 是否决定调用工具
if response.tool_calls:
    print("🔧 LLM 决定调用工具：")
    for call in response.tool_calls:
        print(f"   工具名：{call['name']}")
        print(f"   参数：{call['args']}")
else:
    print("💬 LLM 直接回复：", response.content)

print("=" * 50)
```

运行测试：

```bash
python -m my_agent.step6_tools
```

> ✅ 检查点：能看到 LLM 决定调用 `search_weather` 工具，参数是 `{"city": "北京"}`。
>
> 🧠 **理解要点**：这一步 LLM 只是**说它想调用工具**，但还没有真正执行。
> 真正的执行需要一个**循环**（Agent Loop），这就是 Step 7 要做的事。

---

## Step 7：用 LangGraph 构建完整 Agent

**目标**：用 LangGraph 把 "思考 → 调用工具 → 再思考" 这个循环串起来。

### 7.1 先理解 LangGraph 的核心概念

```
LangGraph 把 Agent 建模为一个「状态图」（State Graph）：

    ┌─────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐     需要调用工具     ┌──────────┐
    │  Agent   │ ──────────────────► │  Tools   │
    │ (LLM    │                      │ (执行工具)│
    │  思考)   │ ◄────────────────── │          │
    └────┬─────┘     返回工具结果     └──────────┘
         │
         │ 不需要工具，直接回复
    ┌────▼─────┐
    │   END    │
    └──────────┘
```

### 7.2 写代码

创建文件 `my_agent/step7_graph.py`：

```python
"""
Step 7：用 LangGraph 构建完整 Agent
运行方式：python -m my_agent.step7_graph
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from my_agent.tools.search import search_weather, calculate

load_dotenv()

# 1. 创建 LLM
llm = ChatOpenAI(model="claude-opus-4-6", temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))

# 2. 定义工具列表
tools = [search_weather, calculate]

# 3. 用 LangGraph 创建 ReAct Agent（一行搞定！）
#    ReAct = Reasoning + Acting，最经典的 Agent 模式
agent = create_react_agent(
    model=llm,
    tools=tools,
    # 可选：给 Agent 设定角色
    prompt="你是一个乐于助人的AI助手。请用中文回复用户的问题。如果需要查询天气或计算，请使用提供的工具。",
)

# 4. 运行 Agent
print("=" * 50)
print("🤖 Agent 已启动！输入 'quit' 退出")
print("=" * 50)

while True:
    user_input = input("\n你：")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("再见！👋")
        break

    # 调用 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    # 打印 Agent 的最终回复
    final_message = result["messages"][-1]
    print(f"\n🤖 Agent：{final_message.content}")
```

运行测试：

```bash
python -m my_agent.step7_graph
```

试试这些对话：

```
你：北京天气怎么样？
🤖 Agent：北京今天晴天，25°C，微风。

你：帮我算一下 123 * 456 + 789
🤖 Agent：计算结果是 56877。

你：你好，介绍一下你自己
🤖 Agent：你好！我是一个 AI 助手，可以帮你查天气、做计算...

你：quit
```

> ✅ 检查点：Agent 能自动判断何时调用工具、何时直接回复。
>
> 🧠 **理解要点**：
> - `create_react_agent` 帮你自动构建了 "思考→工具→再思考" 的循环
> - Agent 会自主决策，你不需要手动写 if/else 判断

---

## Step 8：添加对话记忆

**目标**：让 Agent 记住之前的对话内容。

创建文件 `my_agent/step8_memory.py`：

```python
"""
Step 8：带对话记忆的 Agent
运行方式：python -m my_agent.step8_memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from my_agent.tools.search import search_weather, calculate

load_dotenv()

# 1. 创建 LLM
llm = ChatOpenAI(model="claude-opus-4-6", temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))

# 2. 工具
tools = [search_weather, calculate]

# 3. 创建记忆存储（内存版，重启后丢失）
memory = MemorySaver()

# 4. 创建带记忆的 Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个乐于助人的AI助手。请用中文回复。",
    checkpointer=memory,  # 关键：传入记忆组件
)

# 5. 配置 thread_id（同一个 thread_id = 同一个对话）
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

    final_message = result["messages"][-1]
    print(f"\n🤖 Agent：{final_message.content}")
```

运行测试：

```bash
python -m my_agent.step8_memory
```

试试连续对话：

```
你：我叫小明
🤖 Agent：你好小明！

你：我叫什么名字？
🤖 Agent：你叫小明！      ← 它记住了！

你：帮我查一下上海天气
🤖 Agent：上海多云，22°C...

你：那北京呢？
🤖 Agent：北京晴天，25°C...  ← 它理解了"那北京呢"的上下文
```

> ✅ 检查点：Agent 能记住之前的对话内容，支持上下文理解。

---

## Step 9：用 LangGraph CLI 启动服务（可选进阶）

**目标**：把 Agent 变成一个 HTTP API 服务，可以被前端或其他系统调用。

### 9.1 创建 LangGraph 服务入口

创建文件 `my_agent/graph.py`：

```python
"""
LangGraph 服务入口
用于 langgraph dev 命令启动
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from my_agent.tools.search import search_weather, calculate

load_dotenv()

llm = ChatOpenAI(model="claude-opus-4-6", temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))
tools = [search_weather, calculate]
memory = MemorySaver()

# 导出 graph 变量，供 LangGraph CLI 使用
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个乐于助人的AI助手。请用中文回复。",
    checkpointer=memory,
)
```

### 9.2 创建配置文件

在项目根目录创建 `langgraph.json`：

```json
{
  "graphs": {
    "agent": "./my_agent/graph.py:graph"
  },
  "env": ".env"
}
```

### 9.3 启动服务

```bash
langgraph dev
```

启动后访问：
- API 地址：`http://localhost:8123`
- 可视化界面：`http://localhost:8123/docs`（Swagger UI）

> ✅ 检查点：浏览器打开 `http://localhost:8123/docs` 能看到 API 文档。

---

## Step 10：下一步学习方向

恭喜你！🎉 到这里你已经搭建了一个完整的 AI Agent。接下来可以探索：

### 🔧 添加更多工具
```python
@tool
def search_web(query: str) -> str:
    """搜索互联网信息"""
    # 接入 SerpAPI、Tavily 等搜索 API
    pass

@tool
def read_file(file_path: str) -> str:
    """读取本地文件内容"""
    with open(file_path, 'r') as f:
        return f.read()
```

### 🏗️ 构建多 Agent 协作（进阶）
```
研究员 Agent → 写作 Agent → 审核 Agent
     ↑                            │
     └────── 需要修改时返回 ───────┘
```

### 📚 推荐学习资源

| 资源 | 链接 |
|------|------|
| LangGraph 官方文档 | https://langchain-ai.github.io/langgraph/ |
| LangChain 官方文档 | https://python.langchain.com/ |
| LangSmith 追踪平台 | https://smith.langchain.com/ |
| LangGraph 教程视频 | YouTube 搜索 "LangGraph tutorial" |

### 🗂️ 核心概念速查表

| 概念 | 解释 |
|------|------|
| **LLM** | 大语言模型，Agent 的"大脑" |
| **Tool** | 工具，Agent 可以调用的外部能力 |
| **Agent** | 能自主决策调用工具的 AI 程序 |
| **ReAct** | Reasoning + Acting，先思考再行动的模式 |
| **State Graph** | LangGraph 的核心，用图来描述 Agent 的工作流程 |
| **Checkpointer** | 记忆组件，保存对话历史 |
| **Thread** | 一个对话线程，用 thread_id 区分不同对话 |

---

## 常见问题 FAQ

**Q: 我在国内，连不上 OpenAI 怎么办？**
A: 可以使用第三方中转服务，在 `.env` 中设置 `OPENAI_BASE_URL`（注意不是旧版的 `OPENAI_API_BASE`）。

**Q: 能用免费的模型吗？**
A: 可以！用 Ollama 本地运行开源模型：
```bash
pip install langchain-ollama
```
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b")
```

**Q: Tool 的 docstring 有什么用？**
A: 非常重要！LLM 通过 docstring 理解工具的用途，从而决定何时调用。写得越清晰，Agent 越聪明。

**Q: 生产环境用什么存记忆？**
A: 用数据库替代 `MemorySaver`：
```python
from langgraph.checkpoint.postgres import PostgresSaver
memory = PostgresSaver(connection_string="postgresql://...")
```

---

*文档创建时间：2026-04-17*
*技术栈：LangChain + LangGraph + Claude Opus（第三方中转）*

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

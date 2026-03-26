"""
Tool Schema Definitions for Financial Agent System

定义标准化工具接口，供 MCP Server 和 LLM Function Calling 使用。
参考 MCP 标准格式，确保工具可被 LLM 识别和调用。
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(Enum):
    """工具类别"""
    REALTIME_DATA = "realtime_data"      # 实时数据
    FINANCIAL_ANALYSIS = "financial_analysis"  # 财务分析
    NEWS = "news"                         # 新闻资讯
    WEB = "web"                           # 网页搜索


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """工具定义，兼容 MCP 和 Function Calling 格式"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_function_calling_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式。

        Returns:
            符合 OpenAI function calling 规范的字典
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        """
        转换为 MCP 工具格式。

        Returns:
            MCP 工具 schema
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ==================== 预定义工具 Schema ====================

# 实时股价查询
GET_REALTIME_QUOTE = ToolDefinition(
    name="get_realtime_quote",
    description="获取股票的实时价格报价，包括当前价格、开盘价、最高价、最低价、成交量等。适用于查询'现在股价多少'、'当前价格'等实时行情问题。",
    category=ToolCategory.REALTIME_DATA,
    parameters=[
        ToolParameter(
            name="symbol",
            type="string",
            description="股票代码，支持多种格式：\n"
                       "- A股：000001.SZ (深圳)、600000.SH (上海)\n"
                       "- 港股：0700.HK\n"
                       "- 美股：AAPL、MSFT（无需交易所后缀）\n"
                       "- 指数：^HSI (恒生)、^IXIC (纳斯达克)",
            required=True
        )
    ]
)

# 公司基本信息
GET_COMPANY_INFO = ToolDefinition(
    name="get_company_info",
    description="获取上市公司的基本信息，包括公司名称、行业、市值、员工数、主营业务等基本面数据。适用于查询公司概况、主营业务、行业地位等问题。",
    category=ToolCategory.FINANCIAL_ANALYSIS,
    parameters=[
        ToolParameter(
            name="symbol",
            type="string",
            description="股票代码，格式同上",
            required=True
        )
    ]
)

# 财务数据
GET_FINANCIAL_DATA = ToolDefinition(
    name="get_financial_data",
    description="获取公司的财务报表数据，包括收入、利润、资产负债等关键财务指标。适用于分析公司盈利能力、财务状况等问题。",
    category=ToolCategory.FINANCIAL_ANALYSIS,
    parameters=[
        ToolParameter(
            name="symbol",
            type="string",
            description="股票代码",
            required=True
        ),
        ToolParameter(
            name="period",
            type="string",
            description="财务周期",
            required=False,
            default="annual",
            enum=["annual", "quarterly"]
        )
    ]
)

# 历史价格
GET_HISTORICAL_PRICES = ToolDefinition(
    name="get_historical_prices",
    description="获取股票的历史价格数据，支持日线、周线、月线周期。适用于分析股价走势、历史表现等问题。",
    category=ToolCategory.REALTIME_DATA,
    parameters=[
        ToolParameter(
            name="symbol",
            type="string",
            description="股票代码",
            required=True
        ),
        ToolParameter(
            name="period",
            type="string",
            description="时间范围",
            required=False,
            default="1mo",
            enum=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        ),
        ToolParameter(
            name="interval",
            type="string",
            description="数据间隔",
            required=False,
            default="1d",
            enum=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]
        )
    ]
)

# 股票推荐
GET_RECOMMENDATIONS = ToolDefinition(
    name="get_recommendations",
    description="获取分析师对该股票的评级和推荐信息，包括目标价格、评级趋势等。适用于了解机构对该股票的看法。",
    category=ToolCategory.FINANCIAL_ANALYSIS,
    parameters=[
        ToolParameter(
            name="symbol",
            type="string",
            description="股票代码",
            required=True
        )
    ]
)

# 股票筛选器
SCREEN_STOCKS = ToolDefinition(
    name="screen_stocks",
    description="根据财务指标筛选符合条件的股票，如市值范围、PE、股息率等。适用于寻找符合特定条件的投资标的。",
    category=ToolCategory.FINANCIAL_ANALYSIS,
    parameters=[
        ToolParameter(
            name="criteria",
            type="object",
            description="筛选条件对象，包含以下可选字段：\n"
                       "- market_cap_min: 最小市值（亿）\n"
                       "- market_cap_max: 最大市值（亿）\n"
                       "- pe_min: 最小市盈率\n"
                       "- pe_max: 最大市盈率\n"
                       "- dividend_min: 最小股息率\n"
                       "- sector: 行业板块",
            required=True
        )
    ]
)

# 股票比较
COMPARE_STOCKS = ToolDefinition(
    name="compare_stocks",
    description="比较多个股票的财务指标和表现，包括涨跌幅、市盈率、市净率、股息率等。适用于对比分析不同股票。",
    category=ToolCategory.FINANCIAL_ANALYSIS,
    parameters=[
        ToolParameter(
            name="symbols",
            type="array",
            description="股票代码数组，如 [\"000001.SZ\", \"600000.SH\"]",
            required=True
        ),
        ToolParameter(
            name="metrics",
            type="array",
            description="要比较的指标数组，如 [\"price\", \"pe_ratio\", \"dividend_yield\"]",
            required=False,
            default=["price", "change_percent", "pe_ratio", "market_cap"]
        )
    ]
)


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表，管理所有可用工具"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """注册默认工具"""
        tools = [
            GET_REALTIME_QUOTE,
            GET_COMPANY_INFO,
            GET_FINANCIAL_DATA,
            GET_HISTORICAL_PRICES,
            GET_RECOMMENDATIONS,
            SCREEN_STOCKS,
            COMPARE_STOCKS,
        ]
        for tool in tools:
            self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)

    def get_all(self) -> List[ToolDefinition]:
        """获取所有工具"""
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """按类别获取工具"""
        return [t for t in self._tools.values() if t.category == category]

    def to_function_calling_list(self) -> List[Dict[str, Any]]:
        """获取所有工具的 Function Calling 格式列表"""
        return [tool.to_function_calling_schema() for tool in self._tools.values()]

    def to_mcp_list(self) -> List[Dict[str, Any]]:
        """获取所有工具的 MCP 格式列表"""
        return [tool.to_mcp_schema() for tool in self._tools.values()]

    def register(self, tool: ToolDefinition) -> None:
        """注册新工具"""
        self._tools[tool.name] = tool


# 全局工具注册表实例
registry = ToolRegistry()


# ==================== 工具调用结果格式化 ====================

def format_stock_data(data: Dict[str, Any], format_type: str = "brief") -> str:
    """
    格式化股票数据为可读字符串。

    Args:
        data: yfinance 返回的原始数据
        format_type: 格式类型，"brief" 简洁，"full" 完整

    Returns:
        格式化后的字符串
    """
    if not data:
        return "无数据"

    parts = []

    # 基础信息
    if "symbol" in data:
        parts.append(f"股票代码: {data['symbol']}")
    if "companyName" in data:
        parts.append(f"公司名称: {data['companyName']}")

    # 价格信息
    if "currentPrice" in data:
        parts.append(f"当前价格: ¥{data['currentPrice']:.2f}")
    if "previousClose" in data and "currentPrice" in data:
        change = data['currentPrice'] - data['previousClose']
        pct = (change / data['previousClose']) * 100
        sign = "+" if change >= 0 else ""
        parts.append(f"涨跌额: {sign}{change:.2f} ({sign}{pct:.2f}%)")

    # 详细信息
    if format_type == "full":
        if "dayHigh" in data:
            parts.append(f"最高价: ¥{data['dayHigh']:.2f}")
        if "dayLow" in data:
            parts.append(f"最低价: ¥{data['dayLow']:.2f}")
        if "volume" in data:
            parts.append(f"成交量: {data['volume']:,}")
        if "marketCap" in data:
            cap = data['marketCap']
            if cap > 1e12:
                parts.append(f"市值: {cap/1e12:.2f}万亿")
            elif cap > 1e8:
                parts.append(f"市值: {cap/1e8:.2f}亿")

    return "\n".join(parts)

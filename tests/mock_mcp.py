"""
Mock MCP Server for Offline Testing

模拟 yfinance MCP Server 的返回结果，用于在没有网络或没有启动 npx 服务时进行离线调试。

使用方式:
    # 方式 1: 作为独立的 MCP Server 进程运行
    python tests/mock_mcp.py

    # 方式 2: 导入 MockMCPManager 直接使用
    from tests.mock_mcp import MockMCPManager
    mock = MockMCPManager()
    result = mock.call_tool("get_realtime_quote", {"symbol": "600519.SS"})
"""

import json
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


# ==================== Mock 数据 ====================

MOCK_STOCK_DATA = {
    # A股
    "600519.SS": {
        "symbol": "600519.SS",
        "shortName": "贵州茅台",
        "longName": "贵州茅台酒股份有限公司",
        "currentPrice": 1688.00,
        "previousClose": 1702.50,
        "change": -14.50,
        "changePercent": -0.85,
        "openPrice": 1695.00,
        "dayHigh": 1705.80,
        "dayLow": 1680.00,
        "volume": 2850000,
        "averageVolume": 3200000,
        "marketCap": 2120000000000,
        "peRatio": 28.5,
        "dividendYield": 0.015,
        "52WeekHigh": 1850.00,
        "52WeekLow": 1450.00,
        "recommendationKey": "hold",
        "analystTargetPrice": 1750.00,
        "industry": "白酒",
        "sector": "食品饮料",
        "beta": 0.85,
    },
    "000858.SZ": {
        "symbol": "000858.SZ",
        "shortName": "五粮液",
        "longName": "宜宾五粮液股份有限公司",
        "currentPrice": 145.60,
        "previousClose": 143.20,
        "change": 2.40,
        "changePercent": 1.68,
        "openPrice": 144.00,
        "dayHigh": 146.50,
        "dayLow": 143.00,
        "volume": 15600000,
        "averageVolume": 18000000,
        "marketCap": 565000000000,
        "peRatio": 18.2,
        "dividendYield": 0.028,
        "52WeekHigh": 165.00,
        "52WeekLow": 118.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 160.00,
        "industry": "白酒",
        "sector": "食品饮料",
        "beta": 0.92,
    },
    "601318.SS": {
        "symbol": "601318.SS",
        "shortName": "中国平安",
        "longName": "中国平安保险(集团)股份有限公司",
        "currentPrice": 42.85,
        "previousClose": 43.20,
        "change": -0.35,
        "changePercent": -0.81,
        "openPrice": 43.00,
        "dayHigh": 43.50,
        "dayLow": 42.50,
        "volume": 35600000,
        "averageVolume": 42000000,
        "marketCap": 780000000000,
        "peRatio": 8.5,
        "dividendYield": 0.045,
        "52WeekHigh": 52.00,
        "52WeekLow": 38.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 48.00,
        "industry": "保险",
        "sector": "金融",
        "beta": 1.15,
    },
    "000001.SZ": {
        "symbol": "000001.SZ",
        "shortName": "平安银行",
        "longName": "平安银行股份有限公司",
        "currentPrice": 11.25,
        "previousClose": 11.10,
        "change": 0.15,
        "changePercent": 1.35,
        "openPrice": 11.15,
        "dayHigh": 11.35,
        "dayLow": 11.08,
        "volume": 45000000,
        "averageVolume": 52000000,
        "marketCap": 218000000000,
        "peRatio": 5.2,
        "dividendYield": 0.055,
        "52WeekHigh": 13.50,
        "52WeekLow": 10.00,
        "recommendationKey": "hold",
        "analystTargetPrice": 12.00,
        "industry": "银行",
        "sector": "金融",
        "beta": 1.08,
    },
    "600036.SS": {
        "symbol": "600036.SS",
        "shortName": "招商银行",
        "longName": "招商银行股份有限公司",
        "currentPrice": 35.80,
        "previousClose": 35.50,
        "change": 0.30,
        "changePercent": 0.85,
        "openPrice": 35.60,
        "dayHigh": 36.20,
        "dayLow": 35.40,
        "volume": 28000000,
        "averageVolume": 32000000,
        "marketCap": 920000000000,
        "peRatio": 6.8,
        "dividendYield": 0.038,
        "52WeekHigh": 42.00,
        "52WeekLow": 30.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 40.00,
        "industry": "银行",
        "sector": "金融",
        "beta": 1.02,
    },
    # 港股
    "0700.HK": {
        "symbol": "0700.HK",
        "shortName": "腾讯控股",
        "longName": "腾讯控股有限公司",
        "currentPrice": 298.50,
        "previousClose": 295.00,
        "change": 3.50,
        "changePercent": 1.19,
        "openPrice": 296.00,
        "dayHigh": 300.00,
        "dayLow": 294.50,
        "volume": 12500000,
        "averageVolume": 15000000,
        "marketCap": 2780000000000,
        "peRatio": 22.5,
        "dividendYield": 0.008,
        "52WeekHigh": 380.00,
        "52WeekLow": 260.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 340.00,
        "industry": "互联网",
        "sector": "科技",
        "beta": 1.25,
    },
    # 美股
    "AAPL": {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "longName": "Apple Inc.",
        "currentPrice": 178.50,
        "previousClose": 176.80,
        "change": 1.70,
        "changePercent": 0.96,
        "openPrice": 177.00,
        "dayHigh": 179.50,
        "dayLow": 176.50,
        "volume": 58000000,
        "averageVolume": 65000000,
        "marketCap": 2750000000000,
        "peRatio": 28.0,
        "dividendYield": 0.005,
        "52WeekHigh": 198.00,
        "52WeekLow": 165.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 195.00,
        "industry": "消费电子",
        "sector": "科技",
        "beta": 1.20,
    },
    "MSFT": {
        "symbol": "MSFT",
        "shortName": "Microsoft Corp.",
        "longName": "Microsoft Corporation",
        "currentPrice": 378.50,
        "previousClose": 375.00,
        "change": 3.50,
        "changePercent": 0.93,
        "openPrice": 376.00,
        "dayHigh": 380.00,
        "dayLow": 374.50,
        "volume": 22000000,
        "averageVolume": 25000000,
        "marketCap": 2810000000000,
        "peRatio": 32.0,
        "dividendYield": 0.007,
        "52WeekHigh": 420.00,
        "52WeekLow": 320.00,
        "recommendationKey": "buy",
        "analystTargetPrice": 400.00,
        "industry": "软件",
        "sector": "科技",
        "beta": 0.95,
    },
}


MOCK_FINANCIAL_DATA = {
    "600519.SS": {
        "symbol": "600519.SS",
        "totalRevenue": 147200000000,
        "netIncome": 74700000000,
        "grossMargins": 0.917,
        "operatingMargins": 0.735,
        "profitMargins": 0.528,
        "ebitda": 105000000000,
        "totalAssets": 254000000000,
        "totalLiabilities": 135000000000,
        "shareholdersEquity": 119000000000,
        "debtToEquity": 0.15,
        "returnOnEquity": 0.628,
        "revenuePerShare": 117.25,
        "earningsPerShare": 59.50,
        "bookValuePerShare": 94.75,
        "freeCashFlow": 45000000000,
        "operatingCashFlow": 65000000000,
    },
}


# ==================== Mock MCP Manager ====================

@dataclass
class MockToolResult:
    """模拟工具返回结果"""
    type: str = "text"
    text: str = ""


class MockMCPManager:
    """
    Mock MCP 管理器，用于离线测试。

    模拟 MCP Server 的行为，返回预设的模拟数据。
    """

    def __init__(self):
        self._tools = self._discover_tools()
        self._stock_data = MOCK_STOCK_DATA
        self._financial_data = MOCK_FINANCIAL_DATA

    def _discover_tools(self) -> List[Dict[str, Any]]:
        """返回可用工具列表"""
        return [
            {
                "name": "get_realtime_quote",
                "description": "获取股票的实时价格报价",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_company_info",
                "description": "获取公司基本信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_financial_data",
                "description": "获取财务报表数据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码"
                        },
                        "period": {
                            "type": "string",
                            "description": "财务周期",
                            "enum": ["annual", "quarterly"]
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_historical_prices",
                "description": "获取历史价格",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码"
                        },
                        "period": {
                            "type": "string",
                            "description": "时间范围"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]

    def list_tools(self) -> List[Dict[str, str]]:
        """返回工具描述列表"""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self._tools
        ]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用模拟工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            MCP 格式的工具返回结果
        """
        symbol = arguments.get("symbol", "").upper()

        # 处理不同的股票代码格式
        if symbol in self._stock_data:
            normalized_symbol = symbol
        elif symbol.endswith(".SH") or symbol.endswith(".SZ"):
            # 已经是完整格式
            normalized_symbol = symbol
        elif symbol.startswith("6"):
            normalized_symbol = f"{symbol}.SS"
        elif symbol.startswith(("0", "3")):
            normalized_symbol = f"{symbol}.SZ"
        else:
            normalized_symbol = symbol

        if tool_name == "get_realtime_quote":
            return self._get_realtime_quote(normalized_symbol)
        elif tool_name == "get_company_info":
            return self._get_company_info(normalized_symbol)
        elif tool_name == "get_financial_data":
            return self._get_financial_data(normalized_symbol)
        elif tool_name == "get_historical_prices":
            return self._get_historical_prices(normalized_symbol)
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

    def _get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时报价"""
        if symbol not in self._stock_data:
            # 返回通用模拟数据
            return {
                "success": True,
                "content": [MockToolResult(
                    text=json.dumps({
                        "symbol": symbol,
                        "currentPrice": 100.00,
                        "change": 1.50,
                        "changePercent": 1.52,
                        "message": "Mock 数据 - 未找到该股票"
                    }, ensure_ascii=False)
                )]
            }

        data = self._stock_data[symbol]
        return {
            "success": True,
            "content": [MockToolResult(
                text=json.dumps({
                    "symbol": data["symbol"],
                    "companyName": data["longName"],
                    "currentPrice": data["currentPrice"],
                    "previousClose": data["previousClose"],
                    "change": data["change"],
                    "changePercent": data["changePercent"],
                    "openPrice": data["openPrice"],
                    "dayHigh": data["dayHigh"],
                    "dayLow": data["dayLow"],
                    "volume": data["volume"],
                    "marketCap": data["marketCap"],
                    "trailingPE": data["peRatio"],
                    "recommendationKey": data["recommendationKey"],
                }, ensure_ascii=False)
            )]
        }

    def _get_company_info(self, symbol: str) -> Dict[str, Any]:
        """获取公司信息"""
        if symbol not in self._stock_data:
            return {
                "success": True,
                "content": [MockToolResult(
                    text=json.dumps({"symbol": symbol, "message": "Mock 数据"}, ensure_ascii=False)
                )]
            }

        data = self._stock_data[symbol]
        return {
            "success": True,
            "content": [MockToolResult(
                text=json.dumps({
                    "symbol": data["symbol"],
                    "shortName": data["shortName"],
                    "longName": data["longName"],
                    "industry": data["industry"],
                    "sector": data["sector"],
                    "marketCap": data["marketCap"],
                    "peRatio": data["peRatio"],
                    "dividendYield": data["dividendYield"],
                    "beta": data["beta"],
                }, ensure_ascii=False)
            )]
        }

    def _get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """获取财务数据"""
        if symbol in self._financial_data:
            data = self._financial_data[symbol]
        else:
            # 生成通用模拟财务数据
            data = {
                "symbol": symbol,
                "totalRevenue": 100000000000,
                "netIncome": 10000000000,
                "grossMargins": 0.30,
                "operatingMargins": 0.15,
                "profitMargins": 0.10,
            }

        return {
            "success": True,
            "content": [MockToolResult(
                text=json.dumps(data, ensure_ascii=False)
            )]
        }

    def _get_historical_prices(self, symbol: str) -> Dict[str, Any]:
        """获取历史价格"""
        if symbol not in self._stock_data:
            return {
                "success": True,
                "content": [MockToolResult(
                    text=json.dumps({"symbol": symbol, "message": "Mock 数据"}, ensure_ascii=False)
                )]
            }

        data = self._stock_data[symbol]
        # 生成 30 天模拟历史数据
        import random
        base_price = data["currentPrice"]
        prices = []
        for i in range(30):
            price = base_price * (0.95 + random.random() * 0.1)
            prices.append({
                "date": f"2026-03-{22-i:02d}",
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": int(data["volume"] * (0.8 + random.random() * 0.4))
            })

        return {
            "success": True,
            "content": [MockToolResult(
                text=json.dumps({"symbol": symbol, "prices": prices}, ensure_ascii=False)
            )]
        }


# ==================== MCP Protocol Handler ====================

class MCPProtocolHandler:
    """
    MCP 协议处理器，用于实现 stdio 通信。

    可以作为子进程运行，模拟真实的 MCP Server。
    """

    def __init__(self):
        self.manager = MockMCPManager()
        self._request_id = 0

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method", "")
        request_id = request.get("id", 0)

        if method == "initialize":
            return self._handle_initialize(request_id)
        elif method == "tools/list":
            return self._handle_list_tools(request_id)
        elif method == "tools/call":
            return self._handle_call_tool(request_id, request.get("params", {}))
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

    def _handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """处理 initialize 请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True}
                },
                "serverInfo": {
                    "name": "mock-yfinance",
                    "version": "1.0.0"
                }
            }
        }

    def _handle_list_tools(self, request_id: int) -> Dict[str, Any]:
        """处理 tools/list 请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.manager._discover_tools()
            }
        }

    def _handle_call_tool(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/call 请求"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result = self.manager.call_tool(tool_name, arguments)

        if result.get("success", False):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": result["content"]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": result.get("error", "Unknown error")
                }
            }

    def run(self):
        """运行 MCP Server（stdio 模式）"""
        import sys
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                continue


# ==================== CLI 入口 ====================

if __name__ == "__main__":
    handler = MCPProtocolHandler()
    handler.run()

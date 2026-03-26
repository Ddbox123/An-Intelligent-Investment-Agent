"""
MCP (Model Context Protocol) Client Module

提供异步 MCP Server 连接管理和工具调用能力。
支持多 Server 连接，让 LLM 能够发现并执行外部工具。
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class MCPManager:
    """
    异步 MCP 管理器。

    管理多个 MCP Server 的连接，提供工具发现和调用能力。
    """

    def __init__(self):
        self._sessions: Dict[str, ClientSession] = {}
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._tools: List[MCPTool] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self, servers: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        初始化 MCP Server 连接。

        Args:
            servers: Server 配置字典
                {
                    "yfinance": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-yfinance"]
                    },
                    "fetch": {
                        "command": "python",
                        "args": ["-m", "mcp.server.fetch"]
                    }
                }
        """
        async with self._lock:
            if self._initialized:
                return

            if servers is None:
                servers = self._load_default_servers()

            for name, config in servers.items():
                try:
                    await self._connect_server(name, config)
                    print(f"    [MCP] 已连接 Server: {name}")
                except Exception as e:
                    print(f"    [MCP] 连接 Server '{name}' 失败: {e}")

            await self._discover_tools()
            self._initialized = True

    def _load_default_servers(self) -> Dict[str, Dict[str, Any]]:
        """从 .env 或配置文件加载默认 Server 配置"""
        servers = {}

        # 尝试从环境变量读取
        mcp_servers_env = os.environ.get("MCP_SERVERS", "{}")
        try:
            servers = json.loads(mcp_servers_env)
        except json.JSONDecodeError:
            pass

        return servers

    async def _connect_server(self, name: str, config: Dict[str, Any]) -> None:
        """连接到单个 MCP Server"""
        command = config.get("command", "npx")
        args = config.get("args", [])
        env = config.get("env", {})

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env if env else None
        )

        # 使用 stdio 连接
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            self._sessions[name] = session
            self._servers[name] = config

    async def _discover_tools(self) -> None:
        """从所有已连接的 Server 发现可用工具"""
        self._tools = []

        for name, session in self._sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    self._tools.append(MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        server_name=name
                    ))
            except Exception as e:
                print(f"    [MCP] 发现 Server '{name}' 工具失败: {e}")

    async def list_tools(self) -> List[MCPTool]:
        """
        返回所有可用工具列表。

        Returns:
            MCPTool 对象列表
        """
        return self._tools

    def get_tools_for_llm(self) -> List[Dict[str, str]]:
        """
        获取适合 LLM 使用的工具描述列表。

        Returns:
            包含 name, description 的字典列表
        """
        return [
            {
                "name": tool.name,
                "description": f"[{tool.server_name}] {tool.description}"
            }
            for tool in self._tools
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> ToolCallResult:
        """
        调用指定的 MCP 工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            ToolCallResult 对象
        """
        arguments = arguments or {}

        # 找到工具所在的 Server
        target_session = None
        for tool in self._tools:
            if tool.name == tool_name:
                target_session = self._sessions.get(tool.server_name)
                break

        if not target_session:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"工具 '{tool_name}' 未找到"
            )

        try:
            result = await target_session.call_tool(tool_name, arguments)
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result.content if hasattr(result, 'content') else result
            )
        except Exception as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )

    async def call_tools_parallel(
        self,
        calls: List[Dict[str, Any]]
    ) -> List[ToolCallResult]:
        """
        并行调用多个工具。

        Args:
            calls: [{"name": "tool_name", "arguments": {...}}, ...]

        Returns:
            ToolCallResult 列表
        """
        tasks = []
        for call in calls:
            task = self.call_tool(call["name"], call.get("arguments", {}))
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    def should_use_mcp(self, query: str) -> bool:
        """
        判断查询是否应该使用 MCP 工具。

        Args:
            query: 用户查询

        Returns:
            True 如果查询涉及实时数据/外部信息
        """
        mcp_keywords = [
            "现在", "当前", "实时", "今日", "最新",
            "股价", "股票价格", "价格", "行情",
            "汇率", "货币", "加密货币", "crypto",
            "天气", "新闻", "最新消息",
            "多少", "查询", "获取",
            "今天", "本周", "本月"
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in mcp_keywords)

    def extract_tool_requirements(self, query: str) -> Optional[Dict[str, Any]]:
        """
        从查询中提取工具调用意图。

        Args:
            query: 用户查询

        Returns:
            {"name": "tool_name", "arguments": {...}} 或 None
        """
        # 股价查询模式
        stock_patterns = [
            r"(\w{1,6})股价",
            r"(\w{1,6})的股价",
            r"(\w{1,6})股票价格",
            r"(\w{1,6})现在多少钱",
            r"(\w{1,6})当前价格",
            r"帮我查一下?(\w{1,6})",
            r"(\w{1,6})\s*\(?(?:sz|sh)?\)?",
        ]

        for pattern in stock_patterns:
            match = re.search(pattern, query)
            if match:
                ticker = match.group(1).upper()
                # 添加交易所后缀
                if not ticker.endswith(".SZ") and not ticker.endswith(".SH"):
                    if ticker.startswith("6"):
                        ticker = f"{ticker}.SH"
                    else:
                        ticker = f"{ticker}.SZ"

                return {
                    "name": "get_stock_price",
                    "arguments": {"symbol": ticker}
                }

        # 通用搜索模式
        if "搜索" in query or "查一下" in query or "帮我找" in query:
            return {
                "name": "fetch",
                "arguments": {"url": self._extract_url(query) or query}
            }

        return None

    def _extract_url(self, text: str) -> Optional[str]:
        """从文本中提取 URL"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        return match.group(0) if match else None

    async def close(self) -> None:
        """关闭所有 Server 连接"""
        for name, session in self._sessions.items():
            try:
                await session.close()
            except Exception as e:
                print(f"    [MCP] 关闭 Server '{name}' 失败: {e}")

        self._sessions.clear()
        self._servers.clear()
        self._tools.clear()
        self._initialized = False


# ==================== 同步包装器 ====================

class SyncMCPManager:
    """
    同步包装器，方便在同步代码中使用 MCP。

    使用事件循环在后台运行异步操作。
    """

    def __init__(self):
        self._async_manager = MCPManager()
        self._loop = None

    def initialize(self, servers: Optional[Dict[str, Dict[str, Any]]] = None, timeout: float = 30.0) -> None:
        """同步初始化（带超时）"""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._loop.run_until_complete,
                    self._async_manager.initialize(servers)
                )
                future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"    [MCP] 初始化超时（{timeout}秒），跳过 MCP 连接")
            self._async_manager._initialized = True  # 标记已尝试
        except Exception as e:
            print(f"    [MCP] 初始化异常: {e}")
            self._async_manager._initialized = True

    def list_tools(self) -> List[Dict[str, str]]:
        """同步获取工具列表（带超时）"""
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._loop.run_until_complete,
                    self._async_manager.list_tools()
                )
                return future.result(timeout=10.0)
        except Exception:
            return self._async_manager.get_tools_for_llm()

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0
    ) -> ToolCallResult:
        """同步调用工具（带超时）"""
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._loop.run_until_complete,
                    self._async_manager.call_tool(tool_name, arguments)
                )
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="工具调用超时"
            )
        except Exception as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )

    def get_tools_for_llm(self) -> List[Dict[str, str]]:
        """同步获取 LLM 工具描述"""
        return self._async_manager.get_tools_for_llm()

    def should_use_mcp(self, query: str) -> bool:
        """同步判断是否使用 MCP"""
        return self._async_manager.should_use_mcp(query)

    def extract_tool_requirements(self, query: str) -> Optional[Dict[str, Any]]:
        """同步提取工具需求"""
        return self._async_manager.extract_tool_requirements(query)

    def close(self) -> None:
        """同步关闭（带超时）"""
        if self._loop:
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._loop.run_until_complete,
                        self._async_manager.close()
                    )
                    future.result(timeout=5.0)
            except Exception:
                pass


# ==================== 上下文融合 ====================

def fuse_contexts(
    rag_context: str,
    mcp_results: List[ToolCallResult]
) -> str:
    """
    融合 RAG 检索结果与 MCP 工具调用结果。

    Args:
        rag_context: RAG 检索返回的上下文
        mcp_results: MCP 工具调用结果列表

    Returns:
        融合后的上下文字符串
    """
    parts = []

    # 添加 RAG 上下文
    if rag_context and rag_context != "没有找到相关的参考资料。":
        parts.append("【文档参考资料】")
        parts.append(rag_context)

    # 添加 MCP 实时数据
    mcp_data = []
    for result in mcp_results:
        if result.success:
            content = result.result
            if isinstance(content, list) and len(content) > 0:
                # 处理 MCP 返回的 content 数组
                for item in content:
                    if hasattr(item, 'text'):
                        mcp_data.append(item.text)
                    else:
                        mcp_data.append(str(item))
            else:
                mcp_data.append(str(content))

    if mcp_data:
        parts.append("【实时数据】")
        parts.extend(mcp_data)

    if not parts:
        return "没有找到相关参考资料。"

    return "\n\n".join(parts)


import os


# ==================== Mock MCP Manager 集成 ====================

def get_mock_manager():
    """
    获取 Mock MCP Manager 实例。

    用于离线测试，绕过真实的 MCP Server 连接。
    """
    from tests.mock_mcp import MockMCPManager as Mock

    class MockSyncWrapper:
        """Mock 同步包装器，兼容 SyncMCPManager 接口"""

        def __init__(self):
            self._manager = Mock()

        def list_tools(self):
            return self._manager.list_tools()

        def call_tool(self, tool_name, arguments=None):
            return self._to_tool_call_result(
                self._manager.call_tool(tool_name, arguments or {})
            )

        def get_tools_for_llm(self):
            return [
                {"name": t["name"], "description": t["description"]}
                for t in self._manager.list_tools()
            ]

        def should_use_mcp(self, query):
            return True

        def extract_tool_requirements(self, query):
            import re
            stock_patterns = [
                r"(\w{1,6})股价",
                r"(\w{1,6})的股价",
                r"(\w{1,6})股票价格",
                r"(\w{1,6})现在多少钱",
                r"(\w{1,6})当前价格",
                r"帮我查一下?(\w{1,6})",
            ]
            for pattern in stock_patterns:
                match = re.search(pattern, query)
                if match:
                    ticker = match.group(1).upper()
                    if not ticker.endswith((".SZ", ".SH", ".SS")):
                        if ticker.startswith("6"):
                            ticker = f"{ticker}.SS"
                        else:
                            ticker = f"{ticker}.SZ"
                    return {
                        "name": "get_realtime_quote",
                        "arguments": {"symbol": ticker}
                    }
            return None

        def close(self):
            pass

        def _to_tool_call_result(self, mock_result):
            return ToolCallResult(
                tool_name="unknown",
                success=mock_result.get("success", False),
                result=mock_result.get("content"),
                error=mock_result.get("error")
            )

    return MockSyncWrapper()


# ==================== 上下文融合 ====================

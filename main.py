"""
Main Entry Point for Financial Agentic Workflow System

实现 RAG + 实时工具 的协同调度：

1. Intent Routing (意图路由)
   - Case A (实时类): "现在价格"、"今日涨跌" → 调用 yfinance 工具
   - Case B (分析类): "核心竞争力"、"风险因素" → 启动 RAG 检索
   - Case C (综合类): 两者结合

2. Context Fusion (上下文融合)
   - 将 MCP 实时数字与 RAG 文档片段合并为统一 Prompt

3. System Prompt 升级
   - 强调优先使用 MCP 实时股价进行时效性回答
   - 结合 RAG 资料库进行深度基本面背书
"""

import os
import sys
import json
from typing import List, Optional, Dict, Any, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.config import get_config
from core.retriever import HybridRetriever, RetrievedChunk
from core.mcp_client import MCPManager, SyncMCPManager, ToolCallResult
from core.router import IntentRouter, ContextAggregator, IntentType, IntentResult, AggregatedContext
from core.tools import registry as tool_registry
from src.data_processor import VectorStoreManager


# ==================== System Prompts ====================

# 实时类问题专用提示词
REALTIME_PROMPT = """你是一个专业的股票行情分析师。

你有以下实时行情数据，请基于这些数据回答用户问题：

{context}

回答要求：
- 简洁直接地给出当前价格和涨跌情况
- 如有分析师评级或目标价，一并提供
- 使用 Markdown 格式，重要数据加粗
"""

# 分析类问题专用提示词
ANALYSIS_PROMPT = """你是一个资深的股票基本面分析师。

你有以下文档参考资料，请基于这些资料进行深度分析：

{context}

分析要求：
- 结合文档中的具体数据和事实
- 提供有深度的观点和见解
- 指出风险因素和投资亮点
- 使用 Markdown 格式，重要数据加粗
"""

# 综合类问题提示词（核心提示词升级版）
HYBRID_PROMPT = """你是一个专业的股票投资分析师，结合了实时行情数据和深度基本面研究。

## 信息来源

### 实时数据 (来自 MCP 工具)
{realtime_context}

### 文档资料 (来自 RAG 资料库)
{document_context}

## 回答策略

**你必须遵循以下优先级规则：**

1. **时效性优先**：对于股价、价格、涨跌等实时数据问题，**必须以 MCP 提供的实时数据为准**
   - 例如：问"现在价格"，回答应该是 "¥XXX.XX" 而不是引用文档中的历史价格

2. **基本面背书**：结合 RAG 资料库中的年报、研报数据，为实时数据提供支撑或对比
   - 例如："当前价格较52周高点下跌20%，与机构预测的合理估值区间相符"

3. **信息融合**：将实时数据与文档资料有机结合，提供全面分析
   - 实时数据回答"是什么"
   - 文档资料解释"为什么"

## 禁止事项

- **禁止**：用 RAG 中的历史价格回答实时价格问题
- **禁止**：忽略 MCP 实时数据，只基于文档回答时效性问题
- **禁止**：产生文档中没有的数据或幻觉

## 输出格式

请使用 Markdown 格式回答：
- 实时数据用表格或结构化方式呈现
- 分析部分条理清晰
- 重要数据加粗显示
- 注明数据来源和时间

用户问题：{question}
"""


class FinancialAgenticSystem:
    """
    金融 Agentic Workflow 系统。

    核心特点：
    1. LLM 驱动的意图路由
    2. 实时数据与文档资料的上下文融合
    3. 智能选择 RAG / MCP / Hybrid 处理模式
    """

    def __init__(self):
        self.config = get_config()
        self.vector_store = None
        self.retriever: Optional[HybridRetriever] = None
        self.mcp_manager: Optional[SyncMCPManager] = None
        self.intent_router: Optional[IntentRouter] = None
        self.context_aggregator: Optional[ContextAggregator] = None
        self._llm: Optional[ChatOpenAI] = None
        self._initialize()

    def _initialize(self) -> None:
        """初始化所有组件"""
        print("=" * 60)
        print("金融 Agentic Workflow 系统初始化中...")
        print("=" * 60)

        # 1. 初始化 LLM
        print("\n[1/7] 初始化 LLM...")
        self._llm = ChatOpenAI(
            model=self.config.llm_model,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_api_base,
            temperature=0.3,
            streaming=True
        )
        print(f"    LLM 模型: {self.config.llm_model}")

        # 2. 初始化 Embeddings
        print("\n[2/7] 初始化 Embeddings...")
        from src.data_processor import AliyunEmbeddings
        embeddings = AliyunEmbeddings(
            model=self.config.embedding_model,
            api_key=self.config.openai_api_key
        )
        print(f"    Embedding 模型: {self.config.embedding_model}")

        # 3. 加载向量数据库
        print("\n[3/7] 加载向量数据库...")
        vector_manager = VectorStoreManager(
            persist_directory=self.config.vector_db_path,
            embeddings=embeddings
        )

        try:
            self.vector_store = vector_manager.load_vector_store()
            print(f"    向量数据库已加载")
        except (FileNotFoundError, ValueError) as e:
            print(f"    警告: {e}")
            self.vector_store = None
            print("    提示: 请先运行数据处理脚本创建向量数据库。")

        # 4. 构建混合检索器
        print("\n[4/7] 构建混合检索器...")
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            top_k_initial=20,
            top_k_final=5,
            rerank_model=self.config.rerank_model,
            rerank_api_key=self.config.rerank_api_key
        )
        print(f"    检索器就绪")

        # 5. 初始化 MCP Manager
        print("\n[5/7] 初始化 MCP Manager...")
        self._initialize_mcp()

        # 6. 初始化 Intent Router
        print("\n[6/7] 初始化 Intent Router...")
        self.intent_router = IntentRouter(llm=self._llm)
        print("    Intent Router 就绪")

        # 7. 初始化 Context Aggregator
        print("\n[7/7] 初始化 Context Aggregator...")
        self.context_aggregator = ContextAggregator()
        print("    Context Aggregator 就绪")

        print("\n" + "=" * 60)
        print("初始化完成!")
        print("=" * 60)

    def _initialize_mcp(self) -> None:
        """初始化 MCP 连接"""
        # 检查是否启用 Mock 模式
        use_mock = os.environ.get("MCP_USE_MOCK", "").lower() in ("true", "1", "yes")

        if use_mock:
            print("    [Mock 模式] 使用模拟 MCP Server")
            from core.mcp_client import get_mock_manager
            self.mcp_manager = get_mock_manager()
            tools = self.mcp_manager.list_tools()
            print(f"    Mock MCP 已加载，发现 {len(tools)} 个工具")
            return

        try:
            servers = self.config.mcp_servers_config
        except Exception:
            servers = {}

        if not servers:
            print("    提示: 未配置 MCP Servers (设置 MCP_USE_MOCK=true 可启用模拟模式)")
            return

        try:
            self.mcp_manager = SyncMCPManager()
            self.mcp_manager.initialize(servers)
            tools = self.mcp_manager.list_tools()
            print(f"    已连接 MCP Server，发现 {len(tools)} 个工具")
        except Exception as e:
            print(f"    MCP 初始化失败: {e}")
            print("    提示: 可设置 MCP_USE_MOCK=true 使用模拟模式")
            self.mcp_manager = None

    def _retrieve_rag(self, query: str) -> List[RetrievedChunk]:
        """RAG 检索"""
        if not self.retriever:
            return []
        return self.retriever.retrieve(
            query=query,
            build_bm25=False,
            use_rerank=True
        )

    def _call_mcp_tools(
        self,
        tools: List[str],
        stock_symbol: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        调用 MCP 工具

        Args:
            tools: 工具名称列表
            stock_symbol: 股票代码
            query: 原始查询（用于提取股票代码）

        Returns:
            工具调用结果字典
        """
        if not self.mcp_manager:
            return {}

        results = {}

        # 确定股票代码
        if not stock_symbol and query:
            tool_req = self.mcp_manager.extract_tool_requirements(query)
            if tool_req and "arguments" in tool_req:
                stock_symbol = tool_req["arguments"].get("symbol")

        if not stock_symbol:
            return {}

        # 调用各工具
        for tool_name in tools:
            if tool_name == "get_realtime_quote":
                result = self.mcp_manager.call_tool(
                    "get_realtime_quote",
                    {"symbol": stock_symbol}
                )
                if result.success:
                    results["quote"] = self._parse_tool_result(result)

            elif tool_name == "get_company_info":
                result = self.mcp_manager.call_tool(
                    "get_company_info",
                    {"symbol": stock_symbol}
                )
                if result.success:
                    results["company"] = self._parse_tool_result(result)

            elif tool_name == "get_financial_data":
                result = self.mcp_manager.call_tool(
                    "get_financial_data",
                    {"symbol": stock_symbol}
                )
                if result.success:
                    results["financial"] = self._parse_tool_result(result)

        return results

    def _parse_tool_result(self, result: ToolCallResult) -> Dict[str, Any]:
        """解析 MCP 工具返回结果"""
        try:
            # 兼容 Mock 返回的 dict 和真实的 ToolCallResult
            if isinstance(result, dict):
                content = result.get("content")
            else:
                content = result.result

            if isinstance(content, list) and len(content) > 0:
                # 尝试解析第一个 content item
                item = content[0]
                if hasattr(item, 'text'):
                    text = item.text
                    # 尝试 JSON 解析
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw_text": text}
                return {"raw": str(content)}
            elif isinstance(content, dict):
                return content
            else:
                return {"value": str(content)}
        except Exception as e:
            return {"error": str(e)}

    def _format_rag_context(self, chunks: List[RetrievedChunk]) -> str:
        """格式化 RAG 检索结果"""
        if not chunks:
            return "暂无相关文档资料。"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "未知来源")
            ticker = chunk.metadata.get("ticker", "N/A")

            parts.append(
                f"【参考 {i}】来源: {source}, 股票: {ticker}\n"
                f"{chunk.content}"
            )

        return "\n---\n".join(parts)

    def _format_realtime_context(self, data: Dict[str, Any]) -> str:
        """格式化实时数据"""
        if not data:
            return "无实时数据。"

        parts = []

        if "quote" in data:
            q = data["quote"]
            if isinstance(q, dict):
                if "currentPrice" in q:
                    parts.append(f"当前价格: ¥{q['currentPrice']:.2f}")
                if "change" in q and "changePercent" in q:
                    sign = "+" if q['change'] >= 0 else ""
                    parts.append(f"涨跌: {sign}{q['change']:.2f} ({sign}{q['changePercent']:.2f}%)")
                if "marketCap" in q:
                    cap = q['marketCap']
                    if cap > 1e12:
                        parts.append(f"市值: {cap/1e12:.2f}万亿")
                    elif cap > 1e8:
                        parts.append(f"市值: {cap/1e8:.2f}亿")

        if "company" in data:
            c = data["company"]
            if isinstance(c, dict):
                if "longName" in c:
                    parts.append(f"公司全称: {c['longName']}")
                if "industry" in c:
                    parts.append(f"所属行业: {c['industry']}")

        if "financial" in data:
            f = data["financial"]
            if isinstance(f, dict):
                if "totalRevenue" in f:
                    parts.append(f"总收入: {f['totalRevenue']:,}")
                if "netIncome" in f:
                    parts.append(f"净利润: {f['netIncome']:,}")

        return "\n".join(parts) if parts else "无详细信息。"

    def query(self, question: str, verbose: bool = False) -> str:
        """
        处理用户查询 - Agentic Workflow 核心

        流程：
        1. Intent Routing - LLM 判断意图
        2. Route & Execute - 根据意图执行不同路径
        3. Context Fusion - 融合多源上下文
        4. Generate - LLM 生成回答
        """
        if verbose:
            print("\n" + "=" * 50)
            print("[Step 1] 意图分析")
            print("=" * 50)

        # Step 1: Intent Routing
        intent_result, tools = self.intent_router.route(question)

        if verbose:
            print(f"  意图类型: {intent_result.intent.value}")
            print(f"  置信度: {intent_result.confidence:.2f}")
            print(f"  推理过程: {intent_result.reasoning}")
            print(f"  推荐工具: {tools}")
            print(f"  股票代码: {intent_result.stock_symbol}")

        # Step 2: Route & Execute
        if intent_result.intent == IntentType.REALTIME:
            return self._handle_realtime(question, tools, intent_result, verbose)
        elif intent_result.intent == IntentType.ANALYSIS:
            return self._handle_analysis(question, verbose)
        else:  # HYBRID or UNKNOWN
            return self._handle_hybrid(question, tools, intent_result, verbose)

    def _handle_realtime(
        self,
        question: str,
        tools: List[str],
        intent_result: IntentResult,
        verbose: bool
    ) -> str:
        """处理实时类问题"""
        if verbose:
            print("\n" + "=" * 50)
            print("[模式 A] 实时数据查询")
            print("=" * 50)

        # 调用 MCP 工具
        mcp_data = self._call_mcp_tools(
            tools=tools,
            stock_symbol=intent_result.stock_symbol,
            query=question
        )

        if verbose:
            print(f"  MCP 返回: {json.dumps(mcp_data, ensure_ascii=False)[:200]}...")

        # 格式化上下文
        realtime_context = self._format_realtime_context(mcp_data)

        # 生成回答
        prompt = ChatPromptTemplate.from_messages([
            ("system", REALTIME_PROMPT),
            ("human", "{question}")
        ])

        chain = prompt | self._llm | StrOutputParser()
        return chain.invoke({
            "context": realtime_context,
            "question": question
        })

    def _handle_analysis(self, question: str, verbose: bool) -> str:
        """处理分析类问题"""
        if verbose:
            print("\n" + "=" * 50)
            print("[模式 B] 深度分析查询 (RAG)")
            print("=" * 50)

        # RAG 检索
        chunks = self._retrieve_rag(question)

        if verbose:
            print(f"  检索到 {len(chunks)} 条相关文档")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"    [{i}] {chunk.content[:80]}...")

        # 格式化上下文
        document_context = self._format_rag_context(chunks)

        # 生成回答
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPT),
            ("human", "{question}")
        ])

        chain = prompt | self._llm | StrOutputParser()
        return chain.invoke({
            "context": document_context,
            "question": question
        })

    def _handle_hybrid(
        self,
        question: str,
        tools: List[str],
        intent_result: IntentResult,
        verbose: bool
    ) -> str:
        """处理综合类问题 - RAG + MCP 融合"""
        if verbose:
            print("\n" + "=" * 50)
            print("[模式 C] 综合分析 (RAG + MCP)")
            print("=" * 50)

        # 1. RAG 检索
        if verbose:
            print("\n  [C.1] RAG 检索...")
        chunks = self._retrieve_rag(question)

        if verbose:
            print(f"  检索到 {len(chunks)} 条相关文档")

        # 2. MCP 工具调用
        if verbose:
            print("\n  [C.2] MCP 工具调用...")
        mcp_data = self._call_mcp_tools(
            tools=tools,
            stock_symbol=intent_result.stock_symbol,
            query=question
        )

        if verbose:
            print(f"  MCP 返回: {json.dumps(mcp_data, ensure_ascii=False)[:200]}...")

        # 3. 上下文聚合
        if verbose:
            print("\n  [C.3] 上下文融合...")

        aggregated = self.context_aggregator.aggregate(
            rag_chunks=chunks,
            realtime_data=mcp_data.get("quote", {}),
            question=question
        )

        realtime_context = self.context_aggregator.format_for_llm(
            aggregated, emphasis="realtime"
        )

        document_context = aggregated.document_section

        if verbose:
            print(f"  融合上下文: {len(realtime_context)} + {len(document_context)} 字符")

        # 4. 生成回答 - 使用混合提示词
        if verbose:
            print("\n  [C.4] LLM 生成回答...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", HYBRID_PROMPT),
            ("human", "{question}")
        ])

        chain = prompt | self._llm | StrOutputParser()

        return chain.invoke({
            "realtime_context": realtime_context,
            "document_context": document_context,
            "question": question
        })

    def chat(self) -> None:
        """启动 CLI 交互循环"""
        print("\n" + "=" * 60)
        print("  金融 Agentic Workflow 问答系统")
        print("  支持实时股价查询 + 深度基本面分析")
        print("  输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        while True:
            try:
                print("\n")
                question = input("📝 您的问题: ").strip()

                if not question:
                    continue

                if question.lower() in ["quit", "exit", "q", "退出"]:
                    print("\n感谢使用，再见!")
                    break

                print("\n⏳ 正在分析...")
                response = self.query(question, verbose=True)

                print("\n" + "-" * 60)
                print("📊 回答:")
                print("-" * 60)
                print(response)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\n已退出。")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                continue

    def __del__(self):
        """清理资源"""
        if self.mcp_manager:
            try:
                self.mcp_manager.close()
            except Exception:
                pass


def main():
    """主函数"""
    config = get_config()
    config.ensure_directories()

    # 检查向量数据库
    vector_db_path = config.resolved_vector_db_path
    if not vector_db_path.exists() or not list(vector_db_path.glob("*.sqlite3")):
        print("⚠️  警告: 向量数据库为空或不存在。")
        print("    请先运行 PDF 处理脚本创建向量数据库。")
        print()

    # 启动系统
    agent = FinancialAgenticSystem()
    agent.chat()


if __name__ == "__main__":
    main()

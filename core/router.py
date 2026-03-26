"""
Intent Router and Context Aggregator for Agentic Workflow

实现意图路由和上下文融合两大核心组件：
1. IntentRouter - LLM 驱动的意图分类，决定使用 RAG 还是 MCP 工具
2. ContextAggregator - 将实时数据与文档片段融合为统一上下文
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_config
from core.retriever import RetrievedChunk


class IntentType(Enum):
    """用户意图类型"""
    REALTIME = "realtime"           # 实时数据类（股价、行情）
    ANALYSIS = "analysis"           # 分析类（基本面、风险）
    HYBRID = "hybrid"               # 综合类（两者都需要）
    UNKNOWN = "unknown"             # 未知


@dataclass
class IntentResult:
    """意图分析结果"""
    intent: IntentType
    confidence: float
    reasoning: str
    suggested_tools: List[str]
    stock_symbol: Optional[str] = None


@dataclass
class AggregatedContext:
    """聚合后的上下文"""
    realtime_section: str = ""      # 实时数据区块
    document_section: str = ""      # 文档资料区块
    combined: str = ""             # 融合后的完整上下文
    sources: List[str] = field(default_factory=list)


# ==================== Intent Router ====================

INTENT_CLASSIFICATION_PROMPT = """你是一个专业的金融问题分类器。请分析用户问题，判断其意图类型。

## 意图类型定义

1. **realtime (实时类)**: 用户询问当前或近期的具体数据
   - "现在股价多少"、"今日涨跌"、"当前价格"
   - "今天的成交量"、"当前市值"
   - "实时行情"、"最新报价"

2. **analysis (分析类)**: 用户询问深度分析、原因解释、趋势预测
   - "核心竞争力分析"、"风险因素"
   - "为什么涨/跌"、"未来走势如何"
   - "财务状况评估"、"投资价值"
   - "公司基本面"、"业绩解读"

3. **hybrid (综合类)**: 需要结合实时数据和深度分析的复合问题
   - "结合当前股价分析估值"
   - "现在的价格加上基本面支持吗"
   - "考虑当前行情的投资建议"
   - "现在入手合适吗，结合基本面分析"

4. **unknown (未知)**: 无法明确分类的问题

## 输出格式

请严格按以下 JSON 格式输出，不要包含其他内容：
{
    "intent": "realtime|analysis|hybrid|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "分析理由（1-2句话）",
    "suggested_tools": ["工具名列表，如为空则表示不需要工具"],
    "stock_symbol": "如果问题涉及具体股票，提取股票代码；否则为null"
}

## 用户问题

{question}"""


TOOL_SELECTION_PROMPT = """基于以下意图分析结果，选择合适的 MCP 工具。

## 意图分析
- 意图类型: {intent_type}
- 置信度: {confidence}
- 股票代码: {stock_symbol}

## 可用工具
- get_realtime_quote: 获取实时股价
- get_company_info: 获取公司基本面
- get_financial_data: 获取财务数据
- get_historical_prices: 获取历史价格
- get_recommendations: 获取分析师评级

## 输出格式
JSON数组，如 ["get_realtime_quote"] 或 ["get_realtime_quote", "get_company_info"]
"""


class IntentRouter:
    """
    LLM 驱动的意图路由器。

    使用 LLM 分析用户问题，判断应该使用 RAG、MCP 工具还是两者结合。
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.config = get_config()
        self.llm = llm or self._create_llm()
        self._intent_chain = self._build_intent_chain()
        self._tool_chain = self._build_tool_chain()

    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        return ChatOpenAI(
            model=self.config.llm_model,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_api_base,
            temperature=0.1
        )

    def _build_intent_chain(self):
        """构建意图分类 Chain"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的金融问题分类器。请严格按 JSON 格式输出。"),
            ("human", INTENT_CLASSIFICATION_PROMPT)
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_tool_chain(self):
        """构建工具选择 Chain"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "基于意图分析返回 JSON 数组格式的工具选择。"),
            ("human", TOOL_SELECTION_PROMPT)
        ])
        return prompt | self.llm | StrOutputParser()

    def classify(self, question: str) -> IntentResult:
        """
        分析用户问题，返回意图类型。

        Args:
            question: 用户问题

        Returns:
            IntentResult 对象
        """
        try:
            response = self._intent_chain.invoke({
                "question": question
            })

            # 解析 JSON 响应
            result = json.loads(response)

            intent_str = result.get("intent", "unknown")
            intent = IntentType(intent_str) if intent_str in [e.value for e in IntentType] else IntentType.UNKNOWN

            return IntentResult(
                intent=intent,
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                suggested_tools=result.get("suggested_tools", []),
                stock_symbol=result.get("stock_symbol")
            )

        except Exception as e:
            print(f"    [IntentRouter] 意图分类失败: {e}")
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"分类失败: {e}",
                suggested_tools=[]
            )

    def select_tools(self, intent_result: IntentResult) -> List[str]:
        """
        基于意图选择合适的工具。

        Args:
            intent_result: 意图分析结果

        Returns:
            工具名称列表
        """
        if intent_result.intent == IntentType.REALTIME:
            return ["get_realtime_quote"]
        elif intent_result.intent == IntentType.ANALYSIS:
            return ["get_company_info", "get_financial_data"]
        elif intent_result.intent == IntentType.HYBRID:
            return ["get_realtime_quote", "get_company_info", "get_financial_data"]
        return []

    def route(self, question: str) -> Tuple[IntentResult, List[str]]:
        """
        完整路由：分析意图 + 选择工具。

        Args:
            question: 用户问题

        Returns:
            (意图结果, 工具列表)
        """
        intent_result = self.classify(question)
        tools = self.select_tools(intent_result)
        return intent_result, tools


# ==================== Context Aggregator ====================

class ContextAggregator:
    """
    上下文聚合器。

    将 MCP 返回的实时数据与 RAG 检索到的文档片段融合，
    生成适合 LLM 消费的统一上下文。
    """

    def __init__(self):
        self.config = get_config()

    def aggregate(
        self,
        rag_chunks: List[RetrievedChunk],
        realtime_data: Dict[str, Any],
        question: str
    ) -> AggregatedContext:
        """
        融合 RAG 上下文和实时数据。

        Args:
            rag_chunks: RAG 检索结果
            realtime_data: MCP 工具返回的实时数据
            question: 用户原始问题

        Returns:
            AggregatedContext 对象
        """
        # 1. 格式化实时数据区块
        realtime_section = self._format_realtime_section(realtime_data)

        # 2. 格式化文档区块
        document_section = self._format_document_section(rag_chunks)

        # 3. 生成融合上下文
        combined = self._combine(realtime_section, document_section, question)

        # 4. 收集来源
        sources = []
        if rag_chunks:
            sources.extend([
                chunk.metadata.get("source", "未知文档")
                for chunk in rag_chunks
            ])
        if realtime_data:
            sources.append("MCP 实时数据")

        return AggregatedContext(
            realtime_section=realtime_section,
            document_section=document_section,
            combined=combined,
            sources=list(set(sources))
        )

    def _format_realtime_section(self, data: Dict[str, Any]) -> str:
        """格式化实时数据区块"""
        if not data:
            return ""

        sections = ["【实时数据】"]

        # 股票代码
        if "symbol" in data:
            sections.append(f"股票代码: {data['symbol']}")

        # 公司名称
        if "companyName" in data:
            sections.append(f"公司: {data['companyName']}")

        # 当前价格
        if "currentPrice" in data:
            sections.append(f"当前价格: ¥{data['currentPrice']:.2f}")

        # 涨跌
        if "change" in data and "changePercent" in data:
            sign = "+" if data['change'] >= 0 else ""
            sections.append(
                f"涨跌: {sign}{data['change']:.2f} ({sign}{data['changePercent']:.2f}%)"
            )

        # 开盘价
        if "openPrice" in data:
            sections.append(f"开盘价: ¥{data['openPrice']:.2f}")

        # 最高/低价
        if "dayHigh" in data:
            sections.append(f"最高价: ¥{data['dayHigh']:.2f}")
        if "dayLow" in data:
            sections.append(f"最低价: ¥{data['dayLow']:.2f}")

        # 成交量
        if "volume" in data:
            sections.append(f"成交量: {data['volume']:,}")

        # 市值
        if "marketCap" in data:
            cap = data['marketCap']
            if cap > 1e12:
                sections.append(f"市值: {cap/1e12:.2f}万亿")
            elif cap > 1e8:
                sections.append(f"市值: {cap/1e8:.2f}亿")

        # PE
        if "trailingPE" in data:
            sections.append(f"市盈率(TTM): {data['trailingPE']:.2f}")

        # 分析师评级
        if "recommendationKey" in data:
            sections.append(f"评级: {data['recommendationKey']}")

        return "\n".join(sections)

    def _format_document_section(self, chunks: List[RetrievedChunk]) -> str:
        """格式化文档资料区块"""
        if not chunks:
            return "【文档资料】\n暂无相关文档资料"

        sections = ["【文档参考资料】"]
        sections.append(f"(共找到 {len(chunks)} 条相关资料)")

        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "未知来源")
            ticker = chunk.metadata.get("ticker", "N/A")
            page = chunk.metadata.get("page", "N/A")

            sections.append(
                f"\n参考 {i} (来源: {source}, 股票: {ticker}, 页码: {page})\n"
                f"{chunk.content}"
            )

        return "\n".join(sections)

    def _combine(
        self,
        realtime_section: str,
        document_section: str,
        question: str
    ) -> str:
        """
        生成融合后的完整上下文。

        根据问题类型决定实时数据和文档资料的优先级。
        """
        parts = []

        # 开头：问题摘要
        parts.append(f"用户问题: {question}\n")

        # 实时数据（如果存在）
        if realtime_section:
            parts.append(realtime_section)
            parts.append("")

        # 文档资料
        parts.append(document_section)

        return "\n".join(parts)

    def format_for_llm(
        self,
        aggregated: AggregatedContext,
        emphasis: str = "realtime"
    ) -> str:
        """
        格式化聚合上下文，根据强调类型调整输出。

        Args:
            aggregated: 聚合后的上下文
            emphasis: 强调类型，"realtime" 强调实时数据，"document" 强调文档

        Returns:
            格式化后的字符串
        """
        if emphasis == "realtime" and aggregated.realtime_section:
            return (
                f"{aggregated.realtime_section}\n\n"
                f"{aggregated.document_section}"
            )
        else:
            return aggregated.combined

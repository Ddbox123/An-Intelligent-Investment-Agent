"""
Hybrid Retriever Module with Reranking

Implements hybrid search combining vector similarity and BM25 keyword matching,
with online Rerank API integration for result optimization.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from core.config import config


@dataclass
class RetrievedChunk:
    """结构化检索结果，包含原文和分数。"""
    content: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


class HybridRetriever:
    """
    混合检索器：结合向量检索、BM25 关键词检索和线上 Rerank 重排。

    检索流程：
    1. 向量检索获取候选片段
    2. BM25 关键词检索获取候选片段
    3. 合并去重后调用线上 Rerank API
    4. 返回重排后的 top-k 结果
    """

    def __init__(
        self,
        vector_store: Any,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        rerank_model: Optional[str] = None,
        rerank_api_key: Optional[str] = None
    ):
        """
        初始化混合检索器。

        Args:
            vector_store: ChromaDB 向量存储实例
            top_k_initial: 初筛候选数量（默认 20）
            top_k_final: 最终返回数量（默认 5）
            rerank_model: Rerank 模型名（默认使用 config 配置）
            rerank_api_key: Rerank API Key（默认使用 config 配置）
        """
        self.vector_store = vector_store
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final

        self.rerank_model = rerank_model or config.rerank_model
        self.rerank_api_key = rerank_api_key or config.rerank_api_key

        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: Optional[List[Document]] = None

    def _build_bm25_index(self, documents: List[Document]) -> None:
        """
        构建 BM25 索引。

        Args:
            documents: 文档列表
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        tokenized_corpus = [text.split() for text in texts]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_corpus = documents

    def _vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        向量相似度检索。

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (文档, 相似度分数) 列表
        """
        if not self.vector_store:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def _bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索。

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (文档, BM25 分数) 列表
        """
        if not self._bm25_index or not self._bm25_corpus:
            return []

        tokenized_query = query.split()
        scores = self._bm25_index.get_scores(tokenized_query)

        doc_scores = [
            (self._bm25_corpus[i], scores[i])
            for i in range(len(scores))
            if scores[i] > 0
        ]

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:k]

    def _merge_and_deduplicate(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        alpha: float = 0.5
    ) -> List[Document]:
        """
        合并向量检索和 BM25 检索结果。

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            alpha: 向量检索权重 (1-alpha 是 BM25 权重)

        Returns:
            去重后的文档列表
        """
        seen_content = set()
        merged: List[Document] = []

        # 计算归一化分数
        max_vec_score = max(s for _, s in vector_results) if vector_results else 1.0
        max_bm25_score = max(s for _, s in bm25_results) if bm25_results else 1.0

        all_results: List[Tuple[Document, float, str]] = []

        for doc, score in vector_results:
            norm_score = score / max_vec_score if max_vec_score > 0 else 0
            all_results.append((doc, alpha * norm_score, "vector"))

        for doc, score in bm25_results:
            norm_score = score / max_bm25_score if max_bm25_score > 0 else 0
            all_results.append((doc, (1 - alpha) * norm_score, "bm25"))

        # 按分数排序
        all_results.sort(key=lambda x: x[1], reverse=True)

        # 去重
        for doc, score, source in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged.append(doc)

        return merged

    def _call_rerank_api(
        self,
        query: str,
        documents: List[Document]
    ) -> List[RetrievedChunk]:
        """
        调用阿里百炼 Rerank API 进行重排。

        Args:
            query: 查询文本
            documents: 待重排文档列表

        Returns:
            重排后的检索结果列表
        """
        if not documents or not self.rerank_api_key:
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    score=0.0,
                    metadata=doc.metadata
                )
                for doc in documents
            ]

        import http.client
        import json

        contents = [doc.page_content for doc in documents]

        try:
            # 阿里百炼 Rerank API
            conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")
            payload = json.dumps({
                "model": self.rerank_model,
                "input": {
                    "query": query,
                    "documents": contents
                },
                "parameters": {
                    "return_documents": False
                }
            })

            headers = {
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json"
            }

            conn.request("POST", "/api/v1/services/rerank/text-rerank/text-rerank", payload, headers)
            response = conn.getresponse()
            result = json.loads(response.read().decode())

            if "output" in result and "results" in result["output"]:
                results = result["output"]["results"]
                reranked_chunks = []

                for item in results:
                    # 兼容不同的字段名
                    idx = item.get("documents_index") or item.get("index") or item.get("doc_index")
                    score = item.get("relevance_score") or item.get("score") or item.get("relevance")

                    if idx is not None and idx < len(documents):
                        reranked_chunks.append(RetrievedChunk(
                            content=documents[idx].page_content,
                            score=score if score else 0.0,
                            metadata=documents[idx].metadata
                        ))

                return reranked_chunks
            else:
                print(f"Rerank API 返回格式: {result}")
                raise ValueError("Rerank API 返回格式异常")

        except Exception as e:
            print(f"Rerank API 调用失败: {e}")
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    score=1.0 / (i + 1),
                    metadata=doc.metadata
                )
                for i, doc in enumerate(documents[:self.top_k_final])
            ]

    def retrieve(
        self,
        query: str,
        build_bm25: bool = True,
        use_rerank: bool = True,
        alpha: float = 0.6
    ) -> List[RetrievedChunk]:
        """
        执行混合检索 + 重排。

        Args:
            query: 查询文本
            build_bm25: 是否构建 BM25 索引
            use_rerank: 是否使用 Rerank 重排
            alpha: 向量检索权重

        Returns:
            重排后的检索结果列表
        """
        # 1. 向量检索
        vector_results = self._vector_search(query, k=self.top_k_initial)

        # 2. BM25 检索（如果索引已构建）
        bm25_results: List[Tuple[Document, float]] = []
        if build_bm25 and self._bm25_index:
            bm25_results = self._bm25_search(query, k=self.top_k_initial)

        # 3. 合并去重
        if bm25_results:
            candidates = self._merge_and_deduplicate(vector_results, bm25_results, alpha)
        else:
            candidates = [doc for doc, _ in vector_results]

        # 限制候选数量
        candidates = candidates[:self.top_k_initial]

        if not candidates:
            return []

        # 4. Rerank 重排
        if use_rerank:
            reranked = self._call_rerank_api(query, candidates)
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:self.top_k_final]
        else:
            return [
                RetrievedChunk(
                    content=doc.page_content,
                    score=1.0 / (i + 1),
                    metadata=doc.metadata
                )
                for i, doc in enumerate(candidates[:self.top_k_final])
            ]

    def index_documents(self, documents: List[Document]) -> None:
        """
        为文档构建 BM25 索引。

        Args:
            documents: 文档列表
        """
        self._build_bm25_index(documents)


def create_hybrid_retriever(
    vector_store: Any,
    documents: Optional[List[Document]] = None,
    **kwargs
) -> HybridRetriever:
    """
    便捷函数：创建并初始化混合检索器。

    Args:
        vector_store: 向量存储实例
        documents: 可选，用于构建 BM25 索引的文档
        **kwargs: 传递给 HybridRetriever 的其他参数

    Returns:
        初始化的 HybridRetriever 实例
    """
    retriever = HybridRetriever(vector_store=vector_store, **kwargs)

    if documents:
        retriever.index_documents(documents)

    return retriever

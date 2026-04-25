# Financial RAG System - 财务分析检索系统

基于 RAG (Retrieval-Augmented Generation) 的财务分析检索系统，支持多公司年报检索与智能问答。

## 🎯 项目目标

- **MRR@5 > 75%** ✅ 达成 (82.6%)
- **Recall@5 > 75%** ✅ 达成 (88.5%)

---

## 📊 Benchmark 测试结果

### 最终性能指标

| 指标 | 结果 | 目标 | 达成率 |
|------|------|------|--------|
| **MRR@5** | **82.6%** | >75% | ✅ 110.1% |
| **Recall@5** | **88.5%** | >75% | ✅ 118.0% |
| **Top-1 Accuracy** | 79.1% | - | - |
| **负例准确率** | 100% | - | - |
| **总体准确率** | 81.2% | - | - |

### 优化历程

| 日期 | 优化措施 | MRR@5 | Recall@5 | 变化 |
|------|---------|-------|---------|------|
| 初始 | 基础 HybridRetriever | 77.0% | 82.0% | - |
| 0425-AM | 添加 Amazon/Meta | 77.7% | 83.9% | +0.7% / +1.9% |
| 0425-PM | 修复测试集 (移除BYD/CATL假正例) | 81.4% | 87.8% | +3.7% / +3.9% |
| 0425-PM | 构建 BM25 索引 | 81.5% | 87.8% | +0.1% / 0% |
| 0425-PM | Query Routing | **82.6%** | **88.5%** | +1.1% / +0.7% |

---

## 🏢 向量数据库

### 公司覆盖 (10家, 12,773 chunks)

| 公司 | Ticker | Chunks | 数据来源 |
|------|--------|--------|---------|
| 阿里巴巴 | 09988 | 3,494 | SEC 20-F |
| 腾讯 | 00700 | 2,917 | SEC 20-F |
| Meta | META | 1,313 | SEC 10-K |
| 微软 | MSFT | 1,157 | SEC 10-K |
| 特斯拉 | TSLA | 1,040 | SEC 10-K |
| 英伟达 | NVDA | 948 | SEC 10-K |
| 谷歌 | GOOGL | 945 | SEC 10-K |
| 亚马逊 | AMZN | 785 | SEC 10-K |
| 苹果 | AAPL | 162 | PDF |
| 茅台 | 600519 | 12 | PDF |

### 数据来源

- **SEC EDGAR**: 美国上市公司 10-K / 20-F 年报
- **PDF 文件**: 苹果、茅台等

---

## 🔧 技术架构

### HybridRetriever 混合检索器

```
查询输入
    │
    ▼
┌─────────────────────────────────────┐
│       Query Routing (自动路由)        │
├─────────────────────────────────────┤
│ • 实体/数字查询 → BM25 优先          │
│ • 抽象概念查询 → 向量检索优先         │
│ • 普通查询 → 混合检索 (alpha=0.6)    │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│   1. 向量检索 (Aliyun Embeddings)    │
│      top_k_initial = 50              │
├─────────────────────────────────────┤
│   2. BM25 关键词检索                  │
│      top_k_initial = 50              │
├─────────────────────────────────────┤
│   3. 分数合并归一化                   │
│      alpha = 0.6 (向量权重)           │
│      归一化后加权求和                 │
├─────────────────────────────────────┤
│   4. Rerank API 重排 (阿里百炼)       │
│      top_k_final = 5                  │
└───────────────┬─────────────────────┘
                │
                ▼
           输出 Top-5
```

### 核心组件

| 组件 | 技术 | 说明 |
|------|------|------|
| 向量检索 | Aliyun Embeddings | 1024 维向量 |
| 关键词检索 | BM25 | 基于词频的检索 |
| 重排模型 | 阿里百炼 Rerank | API 调用 |
| 向量存储 | ChromaDB | 本地持久化 |

---

## 📝 测试数据集

### 测试集统计

| 类型 | 数量 | 说明 |
|------|------|------|
| 正例查询 | 148 条 | 公司数据存在 |
| 负例查询 | 17 条 | 公司数据不存在 |
| **总计** | **165 条** | - |

### 按公司分布

| 公司 | 查询数 |
|------|--------|
| 苹果 (AAPL) | 17 |
| 腾讯 (00700) | 15 |
| 特斯拉 (TSLA) | 15 |
| 英伟达 (NVDA) | 15 |
| 谷歌 (GOOGL) | 15 |
| 微软 (MSFT) | 15 |
| 亚马逊 (AMZN) | 15 |
| Meta (META) | 15 |
| 茅台 (600519) | 13 |
| 阿里巴巴 (09988) | 13 |

---

## 🚀 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 Benchmark

```bash
python test_rag_benchmark.py
```

### 3. 添加新公司数据

```python
# 下载 SEC 年报
python download_amazon_meta.py

# 添加到向量库
python add_amazon_meta.py
```

### 4. 查询示例

```python
from core.retriever import HybridRetriever
from src.data_processor import VectorStoreManager

vsm = VectorStoreManager(persist_directory='./data/vector_db', use_aliyun=True)
vs = vsm.load_vector_store()
retriever = HybridRetriever(vector_store=vs, top_k_initial=50, top_k_final=5)

# 执行检索
results = retriever.retrieve("亚马逊云计算收入")
for r in results:
    print(f"{r.metadata.get('ticker')}: {r.content[:100]}... score={r.score}")
```

---

## 📁 项目结构

```
analy-fiance/
├── core/
│   ├── retriever.py       # HybridRetriever 混合检索器
│   └── config.py          # 配置管理
├── src/
│   └── data_processor.py  # 数据处理模块
├── data/
│   ├── raw/               # 原始 PDF/HTML 文件
│   └── vector_db/         # ChromaDB 向量数据库
├── Benchmark/
│   ├── datasets/          # 测试数据集
│   └── metrics.py         # 评估指标
├── scripts/               # 工具脚本
├── test_rag_benchmark.py  # Benchmark 主脚本
└── README.md
```

---

## 🔬 关键优化技术

### 1. Query Routing (查询路由)

根据查询类型自动选择检索策略：

```python
def _route_query(self, query):
    if has_numbers or has_company_keywords:
        return 'keyword_first'  # BM25 优先
    elif has_abstract_concept:
        return 'vector_first'  # 向量优先
    else:
        return 'hybrid'        # 混合
```

**效果**: MRR +1.1%, Top-1 +1.4%

### 2. Hybrid Retrieval (混合检索)

结合向量检索和 BM25 关键词检索：

```python
alpha = 0.6  # 向量权重
score = alpha * norm_vector_score + (1-alpha) * norm_bm25_score
```

### 3. Rerank API (重排)

使用阿里百炼 Rerank 模型对候选文档重新排序。

---

## ⚠️ 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| BYD 数据缺失 | 已知 | 无法自动化获取港交所年报 |
| CATL 数据缺失 | 已知 | 无法自动化获取深交所年报 |
| 部分 Chunk 未优化 | 已知 | Overlap chunking 需全量重建 |

---

## 📈 未来优化方向

- [ ] 全量重建 Chunk Overlap 策略
- [ ] 添加更多公司数据 (BYD/CATL 手动获取)
- [ ] Self-RAG 轻量版实现
- [ ] Query Expansion (查询扩展)

---

## 📄 License

MIT License

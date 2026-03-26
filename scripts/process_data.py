"""
数据处理脚本 - 用于创建向量数据库
"""

from src.data_processor import DataProcessor, VectorStoreManager

# 初始化
processor = DataProcessor(chunk_size=800, chunk_overlap=100)

# 处理 PDF
processor.process_pdf("data/raw/H3_AP202603221820686612_1.pdf", ticker="AAPL")

# 查看处理统计
stats = processor.get_document_stats()
print(f"处理了 {stats['total_documents']} 个文档")

# 创建向量数据库并持久化
vector_manager = VectorStoreManager(persist_directory="data/vector_db")
vector_manager.create_vector_store(processor.documents)

print(f"向量数据库已保存到: data/vector_db")

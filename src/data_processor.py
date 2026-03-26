"""
Data Processor Module for Personal Stock Assistant

This module handles PDF document loading, text chunking with metadata preservation,
and optional Markdown table detection for stock-related financial documents.
"""

import os
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class AliyunEmbeddings(Embeddings):
    """
    阿里百炼 embedding 封装类.

    适配阿里百炼 API 的 embedding 接口.
    """

    def __init__(
        self,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 10
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.dimensions = dimensions
        self.batch_size = batch_size

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用阿里百炼 embedding API (自动分批)."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        all_embeddings = []

        # 分批处理，每批最多 batch_size 条
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询."""
        return self._embed([text])[0]


class MarkdownTableDetector:
    """
    Detects and preserves Markdown table formats from text content.
    
    Handles various table formats commonly found in financial reports:
    - Standard Markdown tables with | separators
    - Tables with/without header separators
    - Tables with varying column counts
    """
    
    # Regex pattern for Markdown tables
    TABLE_PATTERN = re.compile(
        r'(\|.+\|[\r\n]+[-:\s|]+[\r\n]+(?:\|.+\|[\r\n]*)+)',
        re.MULTILINE
    )
    
    # Pattern to detect individual table rows
    ROW_PATTERN = re.compile(r'\|[^\n]+\|', re.MULTILINE)
    
    @classmethod
    def is_table_row(cls, text: str) -> bool:
        """
        Check if a line is a valid Markdown table row.
        
        Args:
            text: Line of text to check
            
        Returns:
            True if the text appears to be a table row
        """
        text = text.strip()
        if not text.startswith('|') or not text.endswith('|'):
            return False
        
        # Must have at least 2 cells
        cells = [c.strip() for c in text.split('|') if c.strip()]
        return len(cells) >= 2
    
    @classmethod
    def is_header_separator(cls, text: str) -> bool:
        """
        Check if a line is a Markdown table header separator.
        
        Args:
            text: Line of text to check
            
        Returns:
            True if the text is a separator line (e.g., |---|---|)
        """
        text = text.strip().strip('|')
        # Replace common separator characters and check if only dashes/colons remain
        cleaned = re.sub(r'[-:|\s]', '', text)
        return len(cleaned) == 0 or (text.replace('-', '').replace(':', '').replace(' ', '') == '')
    
    @classmethod
    def detect_tables(cls, text: str) -> List[Dict[str, Any]]:
        """
        Detect all Markdown tables in the given text.
        
        Args:
            text: Text content to scan for tables
            
        Returns:
            List of dictionaries containing table information:
            {
                'full_table': str,      # Complete table as single string
                'start_pos': int,      # Starting character position
                'end_pos': int,        # Ending character position
                'rows': List[str],     # Individual table rows
                'num_rows': int,       # Number of data rows
                'num_cols': int        # Number of columns
            }
        """
        tables = []
        
        for match in cls.TABLE_PATTERN.finditer(text):
            table_text = match.group(0)
            rows = []
            
            for line in table_text.split('\n'):
                line = line.strip()
                if line and (cls.is_table_row(line) or cls.is_header_separator(line)):
                    if cls.is_header_separator(line):
                        continue  # Skip separator lines
                    rows.append(line)
            
            if rows:
                # Count columns from first row
                num_cols = len([c.strip() for c in rows[0].split('|') if c.strip()])
                
                tables.append({
                    'full_table': table_text,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'rows': rows,
                    'num_rows': len(rows),
                    'num_cols': num_cols
                })
        
        return tables
    
    @classmethod
    def format_as_markdown_table(cls, rows: List[List[str]]) -> str:
        """
        Format rows as a proper Markdown table.
        
        Args:
            rows: List of rows, each row is a list of cell values
            
        Returns:
            Formatted Markdown table string
        """
        if not rows:
            return ""
        
        # Calculate column widths
        num_cols = len(rows[0])
        col_widths = [0] * num_cols
        
        for row in rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build table
        lines = []
        
        for idx, row in enumerate(rows):
            cells = []
            for i, cell in enumerate(row):
                if i < num_cols:
                    cells.append(str(cell).ljust(col_widths[i]))
            
            line = '| ' + ' | '.join(cells) + ' |'
            lines.append(line)
            
            # Add separator after header
            if idx == 0:
                separator_cells = ['-' * col_widths[i] for i in range(num_cols)]
                lines.append('| ' + ' | '.join(separator_cells) + ' |')
        
        return '\n'.join(lines)


class DataProcessor:
    """
    Core data processor for PDF document handling.
    
    Provides functionality to:
    - Load PDF documents using PyMuPDF
    - Split text into chunks with configurable size and overlap
    - Preserve critical metadata (ticker, source_file, page_number)
    - Detect and preserve Markdown table formats
    - Create ChromaDB-compatible document collections
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        embeddings_model: str = "all-MiniLM-L6-v2",
        detect_tables: bool = True
    ):
        """
        Initialize the DataProcessor.
        
        Args:
            chunk_size: Maximum size of each text chunk (default: 800)
            chunk_overlap: Overlap between adjacent chunks (default: 100)
            separators: Custom separator characters for text splitting
            embeddings_model: Name of sentence-transformers model to use
            detect_tables: Whether to preserve Markdown table formatting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_model = embeddings_model
        self.detect_tables = detect_tables
        
        # Default separators prioritized for financial documents
        if separators is None:
            separators = [
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                "|",       # Table separators
                " ",       # Word boundaries
                "."        # Sentence boundaries
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            add_start_index=True
        )
        
        self.documents: List[Document] = []
        self.processed_metadata: List[Dict[str, Any]] = []
    
    def load_pdf(self, pdf_path: str, ticker: str) -> List[Tuple[str, int, str]]:
        """
        Load a PDF document and extract text with page information.
        
        Args:
            pdf_path: Path to the PDF file
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            
        Returns:
            List of tuples: (page_text, page_number, source_file)
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the ticker is empty or invalid
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol cannot be empty")
        
        ticker = ticker.strip().upper()
        
        pages_content = []
        source_file = os.path.basename(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                if text.strip():
                    # Preserve tables if detected
                    if self.detect_tables:
                        text = self._process_tables_in_text(text)
                    
                    pages_content.append((text, page_num + 1, source_file))
            
            doc.close()
            
        except Exception as e:
            raise RuntimeError(f"Error loading PDF: {str(e)}")
        
        return pages_content
    
    def _process_tables_in_text(self, text: str) -> str:
        """
        Process text to preserve Markdown table formatting.
        
        Args:
            text: Raw text content from PDF
            
        Returns:
            Text with tables preserved in Markdown format
        """
        tables = MarkdownTableDetector.detect_tables(text)
        
        if not tables:
            return text
        
        # Replace table sections with clean Markdown format
        result = text
        for table in tables:
            # Check if the table is already in Markdown format
            if table['rows']:
                first_row = table['rows'][0]
                if MarkdownTableDetector.is_table_row(first_row):
                    # Table is already Markdown, keep as-is
                    continue
        
        return text
    
    def _extract_tables_from_raw_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract tables from raw PDF text (often without proper formatting).
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Tuple of (processed_text, list of extracted tables)
        """
        lines = text.split('\n')
        tables = []
        current_table = []
        table_start = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Detect table-like patterns (consecutive lines with consistent separators)
            if '  ' in stripped and '|' not in stripped:
                # Try to detect space-separated columns
                cols = [c.strip() for c in stripped.split() if c.strip()]
                if len(cols) >= 2:
                    # Check if previous lines also look like table rows
                    if current_table or (i > 0 and any('  ' in l.strip() for l in lines[max(0, i-3):i])):
                        current_table.append(stripped)
                        if table_start == -1:
                            table_start = i
            
            elif current_table and '|' in stripped:
                # Convert space-separated to pipe format
                cols = [c.strip() for c in stripped.split() if c.strip()]
                if cols:
                    markdown_row = '| ' + ' | '.join(cols) + ' |'
                    current_table.append(markdown_row)
            
            else:
                # End of potential table
                if len(current_table) >= 2:
                    # Determine column count
                    first_cols = len([c for c in current_table[0].split() if c.strip()])
                    if len(current_table) >= first_cols:
                        tables.append({
                            'rows': current_table,
                            'start_line': table_start,
                            'num_cols': first_cols
                        })
                current_table = []
                table_start = -1
        
        # Handle last table
        if len(current_table) >= 2:
            tables.append({
                'rows': current_table,
                'start_line': table_start,
                'num_cols': len([c for c in current_table[0].split() if c.strip()])
            })
        
        return text, tables
    
    def process_pdf(
        self,
        pdf_path: str,
        ticker: str,
        extract_tables: bool = True
    ) -> List[Document]:
        """
        Process a PDF file and create document chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            ticker: Stock ticker symbol
            extract_tables: Whether to attempt table extraction
            
        Returns:
            List of langchain Document objects with metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the ticker is empty
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol cannot be empty")
        
        ticker = ticker.strip().upper()
        source_file = os.path.basename(pdf_path)
        
        # Load PDF content
        pages_content = self.load_pdf(pdf_path, ticker)
        
        self.documents = []
        self.processed_metadata = []
        
        for page_text, page_number, _ in pages_content:
            # Split text into chunks
            if extract_tables:
                page_text, tables = self._extract_tables_from_raw_text(page_text)
                
                # Add table chunks if tables are detected
                for table in tables:
                    table_md = MarkdownTableDetector.format_as_markdown_table(
                        [row.split('|')[1:-1] for row in table['rows'] if '|' in row]
                    )
                    
                    if table_md:
                        # Create a document for the table
                        table_doc = Document(
                            page_content=table_md,
                            metadata={
                                'ticker': ticker,
                                'source_file': source_file,
                                'page_number': page_number,
                                'content_type': 'table',
                                'num_rows': table['num_cols'],
                                'num_cols': table['num_cols']
                            }
                        )
                        self.documents.append(table_doc)
                        self.processed_metadata.append(table_doc.metadata.copy())
            
            # Split remaining text
            chunks = self.text_splitter.split_text(page_text)
            
            for chunk in chunks:
                if chunk.strip():
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'ticker': ticker,
                            'source_file': source_file,
                            'page_number': page_number,
                            'content_type': 'text'
                        }
                    )
                    self.documents.append(doc)
                    self.processed_metadata.append(doc.metadata.copy())
        
        return self.documents
    
    def process_directory(
        self,
        directory_path: str,
        ticker_mapping: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            ticker_mapping: Optional dict mapping filenames to tickers
            
        Returns:
            List of all processed documents
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                
                # Determine ticker
                if ticker_mapping and filename in ticker_mapping:
                    ticker = ticker_mapping[filename]
                else:
                    # Try to extract ticker from filename
                    ticker = os.path.splitext(filename)[0].upper()
                
                try:
                    docs = self.process_pdf(pdf_path, ticker)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        self.documents = all_documents
        return all_documents
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Returns:
            Dictionary containing processing statistics
        """
        if not self.documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'average_chunk_size': 0,
                'tickers': [],
                'sources': [],
                'page_range': (0, 0)
            }
        
        tickers = set()
        sources = set()
        pages = []
        
        for doc in self.documents:
            tickers.add(doc.metadata.get('ticker', 'UNKNOWN'))
            sources.add(doc.metadata.get('source_file', 'UNKNOWN'))
            pages.append(doc.metadata.get('page_number', 0))
        
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        
        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(self.documents) if self.documents else 0,
            'tickers': list(tickers),
            'sources': list(sources),
            'page_range': (min(pages) if pages else 0, max(pages) if pages else 0)
        }
    
    def save_chunks_to_file(self, output_path: str) -> None:
        """
        Save processed chunks to a JSON file for inspection.
        
        Args:
            output_path: Path for the output JSON file
        """
        import json
        
        output_data = {
            'stats': self.get_document_stats(),
            'documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def ensure_directories(base_path: str = "data") -> None:
        """
        Create the required directory structure for the project.
        
        Args:
            base_path: Base directory path (default: "data")
        """
        directories = [
            os.path.join(base_path, "raw"),
            os.path.join(base_path, "vector_db")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")


class VectorStoreManager:
    """
    Manager for ChromaDB vector store operations.

    Provides functionality to:
    - Create and persist vector stores
    - Add document embeddings
    - Load existing vector stores
    - Perform similarity searches
    """

    def __init__(
        self,
        embeddings_model: str = "text-embedding-v4",
        persist_directory: Optional[str] = None,
        use_aliyun: bool = True,
        embeddings: Any = None
    ):
        """
        Initialize the VectorStoreManager.

        Args:
            embeddings_model: Name of embedding model (default: "text-embedding-v4" for 阿里百炼)
            persist_directory: Directory for persistent storage
            use_aliyun: Whether to use 阿里百炼 API (default: True)
            embeddings: Optional pre-configured embeddings instance
        """
        self.embeddings_model = embeddings_model
        self.persist_directory = persist_directory
        self.use_aliyun = use_aliyun
        self.vector_store = None
        self.embeddings = embeddings

    def load_vector_store(self, persist_directory: Optional[str] = None) -> Any:
        """
        Load an existing vector store from disk.

        Args:
            persist_directory: Directory containing the vector store

        Returns:
            ChromaDB vector store instance
        """
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. "
                "Please install: pip install chromadb"
            )

        load_dir = persist_directory or self.persist_directory

        if not load_dir:
            raise ValueError("No persist_directory specified")

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Vector store directory not found: {load_dir}")

        # 使用已有的 embeddings 或创建新的
        if not self.embeddings:
            if self.use_aliyun:
                self.embeddings = self._create_aliyun_embeddings()
            else:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embeddings_model,
                    model_kwargs={'device': 'cpu'}
                )

        self.vector_store = Chroma(
            persist_directory=load_dir,
            embedding_function=self.embeddings
        )

        return self.vector_store

    def get_vector_store(self) -> Any:
        """
        Get the current vector store instance.

        Returns:
            ChromaDB vector store or None if not initialized
        """
        return self.vector_store

    def _create_aliyun_embeddings(self) -> Any:
        """
        Create 阿里百炼 embedding client.

        Returns:
            AliyunEmbeddings instance configured for 阿里百炼
        """
        return AliyunEmbeddings(
            model=self.embeddings_model,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    
    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: Optional[str] = None
    ) -> Any:
        """
        Create a ChromaDB vector store from documents.

        Args:
            documents: List of langchain Document objects
            persist_directory: Directory for storage (uses instance default if not provided)

        Returns:
            ChromaDB vector store instance
        """
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. "
                "Please install: pip install chromadb"
            )

        if not documents:
            raise ValueError("No documents provided to create vector store")

        persist_dir = persist_directory or self.persist_directory

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

        # 根据配置选择嵌入模型
        if self.use_aliyun:
            self.embeddings = self._create_aliyun_embeddings()
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={'device': 'cpu'}
            )

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

        if persist_dir:
            print(f"Vector store saved to: {persist_dir}")

        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call create_vector_store first.")
        
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call create_vector_store first.")
        
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )


# Convenience function for quick usage
def process_stock_documents(
    pdf_paths: List[str],
    tickers: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    persist_directory: str = "data/vector_db"
) -> Tuple[List[Document], Any]:
    """
    Process stock documents and create vector store.
    
    Args:
        pdf_paths: List of PDF file paths
        tickers: List of corresponding ticker symbols
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        persist_directory: Vector store persistence directory
        
    Returns:
        Tuple of (processed documents, vector store)
    """
    if len(pdf_paths) != len(tickers):
        raise ValueError("Number of PDF paths must match number of tickers")
    
    # Ensure directories exist
    DataProcessor.ensure_directories()
    
    # Process all documents
    processor = DataProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_documents = []
    for pdf_path, ticker in zip(pdf_paths, tickers):
        docs = processor.process_pdf(pdf_path, ticker)
        all_documents.extend(docs)
    
    # Create vector store
    vector_manager = VectorStoreManager(
        persist_directory=persist_directory
    )
    
    vector_store = vector_manager.create_vector_store(all_documents)
    
    return all_documents, vector_store


if __name__ == "__main__":
    # Demo usage
    print("Personal Stock Assistant - Data Processor")
    print("=" * 50)
    
    # Ensure required directories exist
    DataProcessor.ensure_directories()
    
    print("\nConfiguration:")
    print(f"  Chunk Size: 800")
    print(f"  Chunk Overlap: 100")
    print(f"  Embeddings Model: text-embedding-v4 (阿里百炼)")
    print(f"\nPlace PDF files in: data/raw/")
    print(f"Vector database will be stored in: data/vector_db/")

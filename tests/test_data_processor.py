"""
Unit Tests for Data Processor Module

Tests the core functionality of the data_processor module including:
- PDF loading
- Text chunking
- Metadata preservation
- Markdown table detection
"""

import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.data_processor import (
    DataProcessor,
    MarkdownTableDetector,
    VectorStoreManager
)


class TestMarkdownTableDetector(unittest.TestCase):
    """Test cases for Markdown table detection and formatting."""
    
    def test_is_table_row_valid(self):
        """Test detection of valid Markdown table rows."""
        self.assertTrue(MarkdownTableDetector.is_table_row("| Header 1 | Header 2 |"))
        self.assertTrue(MarkdownTableDetector.is_table_row("| Cell 1 | Cell 2 | Cell 3 |"))
    
    def test_is_table_row_invalid(self):
        """Test rejection of invalid table rows."""
        self.assertFalse(MarkdownTableDetector.is_table_row("Just some text"))
        self.assertFalse(MarkdownTableDetector.is_table_row("| Missing end"))
        self.assertFalse(MarkdownTableDetector.is_table_row("No pipes here"))
    
    def test_is_header_separator(self):
        """Test detection of Markdown header separators."""
        self.assertTrue(MarkdownTableDetector.is_header_separator("|---|---|"))
        self.assertTrue(MarkdownTableDetector.is_header_separator("| --- | --- |"))
        self.assertTrue(MarkdownTableDetector.is_header_separator("|:---|:---|"))
    
    def test_detect_tables_simple(self):
        """Test detection of simple Markdown tables."""
        text = """
Some introduction text.

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

More text after the table.
"""
        tables = MarkdownTableDetector.detect_tables(text)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]['num_cols'], 2)
        self.assertGreater(tables[0]['num_rows'], 0)
    
    def test_detect_tables_multiple(self):
        """Test detection of multiple tables in text."""
        text = """
| A | B |
|---|---|
| 1 | 2 |

Some text.

| X | Y | Z |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |
"""
        tables = MarkdownTableDetector.detect_tables(text)
        self.assertEqual(len(tables), 2)
    
    def test_format_as_markdown_table(self):
        """Test Markdown table formatting."""
        rows = [
            ['Header 1', 'Header 2', 'Header 3'],
            ['Value 1', 'Value 2', 'Value 3'],
            ['Value 4', 'Value 5', 'Value 6']
        ]
        
        table = MarkdownTableDetector.format_as_markdown_table(rows)
        
        self.assertIn('|', table)
        self.assertIn('Header 1', table)
        self.assertIn('Value 1', table)
        self.assertIn('---', table)  # Separator line


class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor(
            chunk_size=800,
            chunk_overlap=100
        )
    
    def test_init_defaults(self):
        """Test default initialization values."""
        processor = DataProcessor()
        self.assertEqual(processor.chunk_size, 800)
        self.assertEqual(processor.chunk_overlap, 100)
        self.assertTrue(processor.detect_tables)
    
    def test_init_custom_values(self):
        """Test custom initialization values."""
        processor = DataProcessor(
            chunk_size=500,
            chunk_overlap=50,
            detect_tables=False
        )
        self.assertEqual(processor.chunk_size, 500)
        self.assertEqual(processor.chunk_overlap, 50)
        self.assertFalse(processor.detect_tables)
    
    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            DataProcessor.ensure_directories(base_path=tmpdir)
            
            raw_dir = os.path.join(tmpdir, "raw")
            vector_dir = os.path.join(tmpdir, "vector_db")
            
            self.assertTrue(os.path.exists(raw_dir))
            self.assertTrue(os.path.exists(vector_dir))
    
    def test_load_pdf_file_not_found(self):
        """Test error handling for missing PDF files."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_pdf("nonexistent.pdf", "TEST")
    
    def test_load_pdf_empty_ticker(self):
        """Test error handling for empty ticker."""
        with self.assertRaises(ValueError):
            self.processor.load_pdf("dummy.pdf", "")
    
    def test_process_pdf_empty_ticker(self):
        """Test error handling for empty ticker in process_pdf."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.close()
            try:
                with self.assertRaises(ValueError):
                    self.processor.process_pdf(tmp.name, "")
            finally:
                os.unlink(tmp.name)
    
    def test_get_document_stats_empty(self):
        """Test stats for unprocessed documents."""
        stats = self.processor.get_document_stats()
        
        self.assertEqual(stats['total_documents'], 0)
        self.assertEqual(stats['total_characters'], 0)
        self.assertEqual(stats['tickers'], [])
    
    @patch('src.data_processor.fitz')
    def test_load_pdf_mock(self, mock_fitz):
        """Test PDF loading with mocked PyMuPDF."""
        # Setup mock
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text content"
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__len__ = lambda self: 1
        mock_fitz.open.return_value = mock_doc
        
        # Test
        pages = self.processor.load_pdf("test.pdf", "AAPL")
        
        self.assertEqual(len(pages), 1)
        text, page_num, source = pages[0]
        self.assertEqual(text, "Sample text content")
        self.assertEqual(page_num, 1)
        self.assertEqual(source, "test.pdf")
    
    def test_process_directory_nonexistent(self):
        """Test error handling for missing directory."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_directory("/nonexistent/path")


class TestVectorStoreManager(unittest.TestCase):
    """Test cases for the VectorStoreManager class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        manager = VectorStoreManager()
        self.assertEqual(manager.embeddings_model, "all-MiniLM-L6-v2")
        self.assertIsNone(manager.persist_directory)
        self.assertIsNone(manager.vector_store)
    
    def test_init_custom(self):
        """Test custom initialization."""
        manager = VectorStoreManager(
            embeddings_model="sentence-transformers/all-mpnet-base-v2",
            persist_directory="/path/to/db"
        )
        self.assertEqual(manager.embeddings_model, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(manager.persist_directory, "/path/to/db")
    
    def test_create_vector_store_no_documents(self):
        """Test error handling for empty document list."""
        manager = VectorStoreManager()
        
        with self.assertRaises(ValueError):
            manager.create_vector_store([])
    
    def test_similarity_search_no_vector_store(self):
        """Test error handling when vector store not initialized."""
        manager = VectorStoreManager()
        
        with self.assertRaises(RuntimeError):
            manager.similarity_search("test query")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_full_workflow_concept(self):
        """Test the conceptual full workflow."""
        # Create processor
        processor = DataProcessor(
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Check stats after init (empty)
        stats = processor.get_document_stats()
        self.assertEqual(stats['total_documents'], 0)
        
        # Verify directory creation works
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.ensure_directories(base_path=tmpdir)
            
            raw_path = os.path.join(tmpdir, "raw")
            vector_path = os.path.join(tmpdir, "vector_db")
            
            self.assertTrue(os.path.isdir(raw_path))
            self.assertTrue(os.path.isdir(vector_path))


if __name__ == "__main__":
    unittest.main()

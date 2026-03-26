# Personal Stock Assistant

A RAG-based personal stock assistant that helps you analyze stock-related PDF documents using vector embeddings and retrieval-augmented generation.

## Features

- PDF document processing with intelligent chunking
- Automatic preservation of Markdown table formats
- Vector embeddings using sentence-transformers
- ChromaDB vector store for efficient similarity search
- Rich metadata tracking (ticker, source file, page number)

## Project Structure

```
├── data/
│   ├── raw/                    # Store PDF documents here
│   │   └── .gitkeep
│   └── vector_db/              # Vector database storage
│       └── .gitkeep
├── src/
│   └── data_processor.py       # Core PDF processing logic
├── tests/
│   └── test_data_processor.py  # Unit tests
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.data_processor import DataProcessor

# Initialize processor
processor = DataProcessor(
    chunk_size=800,
    chunk_overlap=100,
    embeddings_model="all-MiniLM-L6-v2"
)

# Process a PDF file
processor.process_pdf(
    pdf_path="data/raw/stock_report.pdf",
    ticker="AAPL"
)

# Create vector store
processor.create_vector_store(
    persist_directory="data/vector_db"
)
```

## Configuration

- **Chunk Size**: 800 characters (default)
- **Chunk Overlap**: 100 characters (default)
- **Embeddings Model**: all-MiniLM-L6-v2 (default)

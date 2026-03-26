"""
Configuration Center for Financial RAG System

Unified configuration management using pydantic-settings.
Loads settings from .env file with validation and type safety.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Config(BaseSettings):
    """Unified configuration class for Financial RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ==================== API Configuration ====================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key for LLM calls"
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )

    # ==================== Model Configuration ====================
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model name for chat completion"
    )
    embedding_model: str = Field(
        default="text-embedding-v4",
        description="Embedding model name for vectorization"
    )

    # ==================== Rerank Configuration ====================
    rerank_api_key: str = Field(
        default="",
        description="API Key for reranking service"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Reranking model name"
    )

    # ==================== Vector Database Configuration ====================
    vector_db_path: str = Field(
        default="./data/vector_db",
        description="Path to store vector database files"
    )

    # ==================== MCP (Model Context Protocol) Configuration ====================
    mcp_server_yfinance_command: str = Field(
        default="npx",
        description="Command to run yfinance MCP server"
    )
    mcp_server_yfinance_args: str = Field(
        default='["-y", "@modelcontextprotocol/server-yfinance"]',
        description="Arguments for yfinance MCP server (JSON array format)"
    )
    mcp_enabled: bool = Field(
        default=True,
        description="Enable MCP integration"
    )

    # ==================== Application Settings ====================
    chunk_size: int = Field(
        default=800,
        description="Text chunk size for document splitting"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between adjacent chunks"
    )
    top_k: int = Field(
        default=4,
        description="Number of documents to retrieve"
    )

    @field_validator("openai_api_base", mode="before")
    @classmethod
    def ensure_api_base_trailing_slash(cls, v: str) -> str:
        """Ensure API base URL has trailing slash."""
        return v.rstrip("/") + "/" if v and not v.endswith("/") else v

    @property
    def resolved_vector_db_path(self) -> Path:
        """Get absolute path for vector database."""
        return Path(self.vector_db_path).resolve()

    @property
    def resolved_api_key(self) -> str:
        """Get API key, raising error if not set."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env file")
        return self.openai_api_key

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.resolved_vector_db_path.mkdir(parents=True, exist_ok=True)

    @property
    def mcp_servers_config(self) -> dict:
        """Get MCP servers configuration as a dictionary."""
        import json
        args = json.loads(self.mcp_server_yfinance_args)

        return {
            "yfinance": {
                "command": self.mcp_server_yfinance_command,
                "args": args
            }
        }


@lru_cache()
def get_config() -> Config:
    """
    Get cached Config instance.
    Uses lru_cache for singleton pattern.

    Returns:
        Config: Singleton configuration instance
    """
    return Config()


# Convenience function for quick access
config = get_config()

"""Configuration management for CRS system."""
from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path


class StoreConfig(BaseModel):
    """Configuration for the Store class."""

    persist_dir: Path = Field(
        default=Path("./crs_data"),
        description="Directory for persisting FAISS index and metadata"
    )
    similarity_metric: Literal["l2", "cosine", "inner_product"] = Field(
        default="cosine",
        description="FAISS similarity metric"
    )
    index_name: str = Field(
        default="crs_index",
        description="Name of the FAISS index file"
    )
    chunk_size: int = Field(
        default=1000,
        description="Batch size for adding documents"
    )


class RerankConfig(BaseModel):
    """Configuration for the Rerank class."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    top_k: int = Field(
        default=20,
        description="Number of candidates to retrieve before reranking"
    )
    final_k: int = Field(
        default=5,
        description="Number of results to return after reranking"
    )
    temporal_decay_factor: float = Field(
        default=0.95,
        description="Decay factor for temporal scoring (0-1)"
    )
    metadata_boost_weights: dict = Field(
        default_factory=lambda: {
            "response_sentiment": 0.1,
            "query_sentiment": 0.1,
        },
        description="Weights for metadata-based score boosting"
    )
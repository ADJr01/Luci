"""Reranking module for improving retrieval accuracy."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from .config import RerankConfig


class Rerank:
    """
    Intelligent reranking system that improves document retrieval accuracy.

    Features:
    - Cross-encoder based semantic reranking
    - Temporal decay scoring (favors recent documents)
    - Metadata-based score boosting
    - Context-aware relevance calculation

    Architecture:
    ┌─────────────────────────────────────┐
    │      Rerank Pipeline                │
    ├─────────────────────────────────────┤
    │ 1. Cross-Encoder Scoring            │
    │ 2. Temporal Decay Application       │
    │ 3. Metadata Boost Calculation       │
    │ 4. Score Fusion & Normalization     │
    │ 5. Final Ranking                    │
    └─────────────────────────────────────┘
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        """
        Initialize the Rerank system.

        Args:
            config: Reranking configuration. Uses defaults if None.
        """
        self.config = config or RerankConfig()
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize cross-encoder model for reranking."""
        try:
            self.cross_encoder = CrossEncoder(self.config.model_name)
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder: {e}")
            print("Falling back to score-only reranking")
            self.cross_encoder = None

    def rerank(
            self,
            query: str,
            documents: List[Document],
            initial_scores: Optional[List[float]] = None
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents using multi-factor scoring.

        Args:
            query: The search query
            documents: List of candidate documents
            initial_scores: Optional initial similarity scores from FAISS

        Returns:
            List of (Document, final_score) tuples, sorted by relevance
        """
        if not documents:
            return []

        # 1. Calculate cross-encoder scores
        semantic_scores = self._calculate_semantic_scores(query, documents)

        # 2. Calculate temporal decay scores
        temporal_scores = self._calculate_temporal_scores(documents)

        # 3. Calculate metadata boost scores
        metadata_scores = self._calculate_metadata_scores(documents)

        # 4. Fuse scores
        final_scores = self._fuse_scores(
            semantic_scores,
            temporal_scores,
            metadata_scores,
            initial_scores
        )

        # 5. Sort and return top-k
        ranked_results = [
            (doc, score)
            for doc, score in zip(documents, final_scores)
        ]
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        return ranked_results[:self.config.final_k]

    def _calculate_semantic_scores(
            self,
            query: str,
            documents: List[Document]
    ) -> np.ndarray:
        """Calculate semantic relevance using cross-encoder."""
        if self.cross_encoder is None:
            # Fallback: return uniform scores
            return np.ones(len(documents))

        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.cross_encoder.predict(pairs)
            # Normalize to 0-1 range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return np.array(scores)
        except Exception as e:
            print(f"Warning: Cross-encoder scoring failed: {e}")
            return np.ones(len(documents))

    def _calculate_temporal_scores(
            self,
            documents: List[Document]
    ) -> np.ndarray:
        """
        Apply temporal decay to favor recent documents.

        Score = decay_factor ^ (days_since_response)
        """
        scores = []
        now = datetime.now()

        for doc in documents:
            response_time = doc.metadata.get("response_time")

            if isinstance(response_time, datetime):
                days_diff = (now - response_time).days
                # Apply exponential decay
                score = self.config.temporal_decay_factor ** days_diff
            else:
                score = 1.0  # No temporal information

            scores.append(score)

        return np.array(scores)

    def _calculate_metadata_scores(
            self,
            documents: List[Document]
    ) -> np.ndarray:
        """
        Boost scores based on metadata features.

        Currently supports:
        - Sentiment matching
        - Model preference
        - Source quality
        """
        scores = np.zeros(len(documents))

        for idx, doc in enumerate(documents):
            boost = 0.0

            # Sentiment boost
            if doc.metadata.get("query_sentiment") == "positive":
                boost += self.config.metadata_boost_weights.get(
                    "query_sentiment", 0.1
                )

            if doc.metadata.get("response_sentiment") == "positive":
                boost += self.config.metadata_boost_weights.get(
                    "response_sentiment", 0.1
                )

            scores[idx] = boost

        return scores

    def _fuse_scores(
            self,
            semantic: np.ndarray,
            temporal: np.ndarray,
            metadata: np.ndarray,
            initial: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Fuse multiple score sources using weighted combination.

        Final = 0.5*semantic + 0.2*temporal + 0.1*metadata + 0.2*initial
        """
        # Normalize all scores to 0-1 range
        semantic_norm = semantic / (semantic.max() + 1e-10)
        temporal_norm = temporal / (temporal.max() + 1e-10)
        metadata_norm = metadata / (metadata.max() + 1e-10) if metadata.max() > 0 else metadata

        # Weighted combination
        final_scores = (
                0.5 * semantic_norm +
                0.2 * temporal_norm +
                0.1 * metadata_norm
        )

        # Add initial FAISS scores if available
        if initial is not None:
            initial_array = np.array(initial)
            initial_norm = initial_array / (initial_array.max() + 1e-10)
            final_scores += 0.2 * initial_norm

        return final_scores
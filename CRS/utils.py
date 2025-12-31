"""Utility functions for CRS system."""
from typing import List, Dict, Any
from datetime import datetime
from langchain.schema import Document


def create_chat_document(
        query: str,
        response: str,
        chat_id: str,
        chat_index: int,
        response_sources: List[str],
        response_model: str,
        query_sentiment: str = "",
        response_sentiment: str = "",
        response_toolset: str = ""
) -> Document:
    """
    Factory function for creating standardized chat documents.

    Args:
        query: User query
        response: System response
        chat_id: Unique chat session identifier
        chat_index: Message index in chat session
        response_sources: List of source documents used
        response_model: Model used for generation
        query_sentiment: Sentiment of user query
        response_sentiment: Sentiment of response
        response_toolset: Tools used in response generation

    Returns:
        Formatted Document ready for storage
    """
    return Document(
        page_content=response,
        metadata={
            "chat_id": chat_id,
            "chat_index": chat_index,
            "response_time": datetime.now(),
            "response_source": response_sources,
            "response_toolset": response_toolset,
            "query_sentiment": query_sentiment,
            "response_sentiment": response_sentiment,
            "response_model": response_model,
            "response_sources": ", ".join(response_sources),
            "query": query
        }
    )


def format_results(results: List[tuple[Document, float]]) -> str:
    """
    Format query results for human-readable output.

    Args:
        results: List of (Document, score) tuples

    Returns:
        Formatted string representation
    """
    if not results:
        return "No results found."

    output = []
    output.append(f"\n{'=' * 80}")
    output.append(f"Found {len(results)} results")
    output.append(f"{'=' * 80}\n")

    for idx, (doc, score) in enumerate(results, 1):
        output.append(f"Result #{idx} (Score: {score:.4f})")
        output.append(f"{'-' * 80}")
        output.append(f"Content: {doc.page_content[:200]}...")
        output.append(f"\nMetadata:")
        for key, value in doc.metadata.items():
            output.append(f"  {key}: {value}")
        output.append(f"{'=' * 80}\n")

    return "\n".join(output)
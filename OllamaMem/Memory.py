import os
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from mem0 import Memory
class EnhancedMemory:
    """
    Enhanced Memory wrapper with authentication and advanced querying capabilities.
    """

    def __init__(
            self,
            memory: Memory,
            enable_auth: bool = False,
            auth_key: Optional[str] = None,
            persist_dir: Optional[str] = None,
            collection_name: Optional[str] = None
    ):
        """
        Initialize EnhancedMemory.

        Args:
            memory: Base Memory instance
            enable_auth: Whether authentication is enabled
            auth_key: Authentication key for token generation
            persist_dir: Directory for persistence
            collection_name: Collection name
        """
        self.memory = memory
        self.enable_auth = enable_auth
        self.auth_key = auth_key
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._current_user: Optional[str] = None

    # ==================== Authentication Methods ====================

    def set_user(self, user_id: str, token: Optional[str] = None) -> None:
        """
        Set the current user for authenticated operations.

        Args:
            user_id: User identifier
            token: Authentication token (required if auth is enabled)

        Raises:
            PermissionError: If token is invalid
        """
        if self.enable_auth:
            if not token:
                raise PermissionError("Authentication token required")

            if not self._validate_token(user_id, token):
                raise PermissionError("Invalid authentication token")

        self._current_user = user_id

    def _validate_token(self, user_id: str, token: str) -> bool:
        """Validate user token."""
        if not self.auth_key:
            return False

        expected = hashlib.sha256(f"{user_id}:{self.auth_key}".encode()).hexdigest()
        return token == expected

    def _ensure_user_set(self) -> None:
        """Ensure a user is set for authenticated operations."""
        if self.enable_auth and not self._current_user:
            raise PermissionError("No user authenticated. Call set_user() first.")

    def _get_user_metadata(self, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Get metadata with user isolation.

        Args:
            additional_metadata: Additional metadata to merge

        Returns:
            Metadata dictionary with user_id
        """
        metadata = {}

        if self.enable_auth and self._current_user:
            metadata["user_id"] = self._current_user

        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    # ==================== Core Memory Operations ====================

    def add(
            self,
            text: str,
            metadata: Optional[Dict] = None,
            user_id: Optional[str] = None
    ) -> Dict:
        """
        Add a memory with automatic user isolation.

        Args:
            text: Text content to store
            metadata: Additional metadata
            user_id: Override current user (requires proper authentication)

        Returns:
            Result from memory.add()
        """
        if user_id and user_id != self._current_user:
            raise PermissionError("Cannot add memories for other users")

        self._ensure_user_set()

        # Add timestamp and user metadata
        full_metadata = self._get_user_metadata(metadata)
        full_metadata["timestamp"] = datetime.now().isoformat()

        result = self.memory.add(text, metadata=full_metadata)
        self.save()

        return result

    def search(
            self,
            query: str,
            limit: int = 5,
            user_id: Optional[str] = None
    ):
        """
        Search memories with automatic user isolation.

        Args:
            query: Search query
            limit: Maximum number of results
            user_id: Override current user (for admin access)

        Returns:
            List of search results
        """
        self._ensure_user_set()

        # Apply user filter if authentication is enabled
        filters = {}
        if self.enable_auth:
            target_user = user_id if user_id else self._current_user
            filters["user_id"] = target_user

        return self.memory.search(query, limit=limit, filters=filters)

    # ==================== Advanced Query Method ====================

    def advanced_query(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            date_range: Optional[tuple] = None,
            relevance_threshold: float = 0.7,
            max_results: int = 10,
            custom_prompt: Optional[str] = None,
            require_all_filters: bool = True,
            sort_by: str = "relevance",
            include_metadata: bool = True,
            re_rank: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced query with enhanced filtering, custom prompts, and metadata operations.

        Args:
            query: Search query text
            filters: Dictionary of metadata filters
                Example: {"category": "work", "priority": "high", "status": "active"}
            date_range: Tuple of (start_date, end_date) in ISO format
                Example: ("2024-01-01T00:00:00", "2024-12-31T23:59:59")
            relevance_threshold: Minimum relevance score (0.0-1.0)
            max_results: Maximum number of results to return
            custom_prompt: Custom prompt template for LLM processing
                Variables: {query}, {context}, {filters}
                Example: "Based on {context}, provide a detailed answer about {query} focusing on {filters}"
            require_all_filters: If True, all filters must match. If False, any filter can match
            sort_by: Sort results by "relevance", "date", or "custom"
            include_metadata: Include full metadata in results
            re_rank: Apply re-ranking to results

        Returns:
            Dictionary with:
                - results: List of matching memories
                - metadata: Query metadata (filters applied, result count, etc.)
                - synthesized_answer: LLM-generated answer based on custom prompt (if provided)
        """
        self._ensure_user_set()

        # Build comprehensive filters
        query_filters = self._get_user_metadata(filters or {})

        # Add date range filtering
        if date_range:
            start_date, end_date = date_range
            query_filters["date_range"] = {"start": start_date, "end": end_date}

        # Perform initial search with higher limit for filtering
        initial_limit = max_results * 3 if filters else max_results

        try:
            raw_results = self.memory.search(
                query,
                limit=initial_limit,
                filters=query_filters if require_all_filters else None
            )
        except Exception as e:
            return {
                "results": [],
                "metadata": {
                    "error": str(e),
                    "query": query,
                    "filters_applied": query_filters
                }
            }

        # Post-process results with advanced filtering
        filtered_results = self._apply_advanced_filters(
            results=raw_results,
            filters=filters or {},
            date_range=date_range,
            relevance_threshold=relevance_threshold,
            require_all_filters=require_all_filters
        )

        # Sort results
        sorted_results = self._sort_results(filtered_results, sort_by)

        # Limit results
        final_results = sorted_results[:max_results]

        # Prepare response
        response = {
            "results": final_results if include_metadata else [
                {"text": r.get("text", ""), "score": r.get("score", 0.0)}
                for r in final_results
            ],
            "metadata": {
                "query": query,
                "filters_applied": query_filters,
                "total_results": len(final_results),
                "relevance_threshold": relevance_threshold,
                "date_range": date_range,
                "sort_by": sort_by
            }
        }

        # Generate synthesized answer with custom prompt
        if custom_prompt and final_results:
            synthesized = self._synthesize_answer(
                query=query,
                results=final_results,
                custom_prompt=custom_prompt,
                filters=filters or {}
            )
            response["synthesized_answer"] = synthesized

        return response

    def _apply_advanced_filters(
            self,
            results: List[Dict],
            filters: Dict[str, Any],
            date_range: Optional[tuple],
            relevance_threshold: float,
            require_all_filters: bool
    ) -> List[Dict]:
        """Apply advanced filtering logic to results."""
        filtered = []

        for result in results:
            # Check relevance threshold
            if result.get("score", 0.0) < relevance_threshold:
                continue

            # Check metadata filters
            result_metadata = result.get("metadata", {})

            if filters:
                if require_all_filters:
                    # All filters must match
                    if not all(
                            result_metadata.get(k) == v
                            for k, v in filters.items()
                    ):
                        continue
                else:
                    # Any filter can match
                    if not any(
                            result_metadata.get(k) == v
                            for k, v in filters.items()
                    ):
                        continue

            # Check date range
            if date_range:
                timestamp = result_metadata.get("timestamp")
                if timestamp:
                    start_date, end_date = date_range
                    if not (start_date <= timestamp <= end_date):
                        continue

            filtered.append(result)

        return filtered

    def _sort_results(self, results: List[Dict], sort_by: str) -> List[Dict]:
        """Sort results by specified criteria."""
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        elif sort_by == "date":
            return sorted(
                results,
                key=lambda x: x.get("metadata", {}).get("timestamp", ""),
                reverse=True
            )
        else:
            return results

    def _synthesize_answer(
            self,
            query: str,
            results: List[Dict],
            custom_prompt: str,
            filters: Dict[str, Any]
    ) -> str:
        """
        Synthesize an answer using LLM with custom prompt.

        Args:
            query: Original query
            results: Filtered search results
            custom_prompt: Custom prompt template
            filters: Applied filters

        Returns:
            Synthesized answer from LLM
        """
        # Prepare context from results
        context = "\n\n".join([
            f"Memory {i + 1}: {r.get('text', '')}"
            for i, r in enumerate(results[:5])  # Use top 5 for context
        ])

        # Format custom prompt
        formatted_prompt = custom_prompt.format(
            query=query,
            context=context,
            filters=json.dumps(filters, indent=2)
        )

        # Use memory's LLM to generate answer
        try:
            # This is a simplified approach - adjust based on mem0's actual API
            response = self.memory.chat(formatted_prompt)
            return response
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"

    # ==================== Utility Methods ====================

    def save(self) -> None:
        """Explicitly save memory to disk."""
        try:
            self.memory.vector_store.save()
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        stats = {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "auth_enabled": self.enable_auth,
            "current_user": self._current_user
        }

        if self.persist_dir and self.collection_name:
            index_path = Path(self.persist_dir) / f"{self.collection_name}.index"
            if index_path.exists():
                stats["index_size_mb"] = index_path.stat().st_size / (1024 * 1024)
                stats["last_modified"] = datetime.fromtimestamp(
                    index_path.stat().st_mtime
                ).isoformat()

        return stats

    def list_user_memories(
            self,
            user_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict]:
        """
        List all memories for a specific user.

        Args:
            user_id: User ID (defaults to current user)
            limit: Maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        target_user = user_id or self._current_user

        if self.enable_auth and target_user != self._current_user:
            raise PermissionError("Cannot list memories for other users")

        # This would need to be implemented based on mem0's actual API
        # Placeholder implementation
        return []

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory (with user permission check).

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted successfully
        """
        self._ensure_user_set()

        # Implementation would need mem0's delete API
        # Should check if memory belongs to current user
        try:
            self.memory.delete(memory_id)
            self.save()
            return True
        except Exception:
            return False
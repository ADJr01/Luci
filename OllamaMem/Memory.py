import os
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from mem0 import Memory


class SessionNotSetError(Exception):
    """Exception raised when attempting operations without an active user session."""
    pass


class EnhancedMemory:
    """
    Enhanced Memory wrapper with session management and advanced querying capabilities.
    """

    def __init__(
            self,
            memory: Memory,
            persist_dir: Optional[str] = None,
            collection_name: Optional[str] = None,
            re_ranking_model: Optional[str] = None,
            re_rank_top_k: int = 15,
            enable_reranking: bool = False
    ):
        """
        Initialize EnhancedMemory.

        Args:
            memory: Base Memory instance
            persist_dir: Directory for persistence
            collection_name: Collection name
            re_ranking_model: Re-ranking model (for future use)
            re_rank_top_k: Number of results to re-rank
            enable_reranking: Whether re-ranking is enabled
        """
        self.memory = memory
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._current_user: Optional[str] = None
        self._active_sessions: Dict[str, datetime] = {}

        # Re-ranking config (for future implementation)
        self.re_ranking_model = re_ranking_model
        self.re_rank_top_k = re_rank_top_k
        self.enable_reranking = enable_reranking

    # ==================== Session Management ====================

    def set_user_session(self, user_name: str) -> None:
        """
        Start a session for a specific user.

        Args:
            user_name: User identifier

        Example:
            memory.set_user_session("alice")
            memory.add("Alice's note")
        """
        if not user_name or not isinstance(user_name, str):
            raise ValueError("user_name must be a non-empty string")

        self._current_user = user_name
        self._active_sessions[user_name] = datetime.now()
        print(f"✓ Session started for user: {user_name}")

    def end_user_session(self, user_name: Optional[str] = None) -> None:
        """
        End a session for a specific user.

        Args:
            user_name: User identifier (defaults to current user)

        Example:
            memory.end_user_session("alice")
            # or
            memory.end_user_session()  # ends current user's session
        """
        target_user = user_name if user_name else self._current_user

        if not target_user:
            raise SessionNotSetError(
                "No active session to end. Use set_user_session() first."
            )

        if target_user in self._active_sessions:
            del self._active_sessions[target_user]

        if self._current_user == target_user:
            self._current_user = None

        print(f"✓ Session ended for user: {target_user}")

    def get_current_user(self) -> Optional[str]:
        """
        Get the current active user.

        Returns:
            Current user name or None
        """
        return self._current_user

    def get_active_sessions(self) -> Dict[str, str]:
        """
        Get all active sessions.

        Returns:
            Dictionary of user_name -> session_start_time (ISO format)
        """
        return {
            user: start_time.isoformat()
            for user, start_time in self._active_sessions.items()
        }

    def _ensure_session_active(self) -> None:
        """Ensure a user session is active before operations."""
        if not self._current_user:
            raise SessionNotSetError(
                "No active user session. Call set_user_session(user_name) before performing operations."
            )

    def _get_user_metadata(self, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Get metadata without user_id (since mem0 handles it separately).

        Args:
            additional_metadata: Additional metadata to merge

        Returns:
            Metadata dictionary
        """
        self._ensure_session_active()

        metadata = {}

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
            user_id: Override current user (for admin operations)

        Returns:
            Result from memory.add()

        Raises:
            SessionNotSetError: If no user session is active
        """
        if user_id:
            # Temporarily override user for this operation
            original_user = self._current_user
            self._current_user = user_id
            try:
                return self.add(text, metadata)
            finally:
                self._current_user = original_user

        self._ensure_session_active()

        # Prepare metadata (without user_id, as it's a separate parameter)
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["timestamp"] = datetime.now().isoformat()

        # Pass user_id as a separate parameter to mem0
        result = self.memory.add(text, user_id=self._current_user, metadata=full_metadata)
        self.save()

        return result

    def search(
            self,
            query: str,
            limit: int = 5,
            user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search memories with automatic user isolation.

        Args:
            query: Search query
            limit: Maximum number of results
            user_id: Override current user (for cross-user search if needed)

        Returns:
            List of search results

        Raises:
            SessionNotSetError: If no user session is active
        """
        if user_id:
            # Temporarily override user for this operation
            original_user = self._current_user
            self._current_user = user_id
            try:
                return self.search(query, limit)
            finally:
                self._current_user = original_user

        self._ensure_session_active()

        # Pass user_id as parameter to mem0, not in filters
        try:
            results = self.memory.search(query, user_id=self._current_user, limit=limit)
            return results if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_all(
            self,
            limit: int = 100,
            user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all memories for the current user.

        Args:
            limit: Maximum number of memories to return
            user_id: Override current user

        Returns:
            List of all memories

        Raises:
            SessionNotSetError: If no user session is active
        """
        if user_id:
            original_user = self._current_user
            self._current_user = user_id
            try:
                return self.get_all(limit)
            finally:
                self._current_user = original_user

        self._ensure_session_active()

        # Get all memories using user_id parameter
        try:
            all_memories = self.memory.get_all(user_id=self._current_user)
            return all_memories[:limit] if all_memories else []
        except Exception as e:
            print(f"Error getting memories: {e}")
            return []

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
            user_id: Optional[str] = None
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
                Example: "Based on {context}, provide a detailed answer about {query}"
            require_all_filters: If True, all filters must match. If False, any filter can match
            sort_by: Sort results by "relevance", "date", or "custom"
            include_metadata: Include full metadata in results
            user_id: Override current user

        Returns:
            Dictionary with:
                - results: List of matching memories
                - metadata: Query metadata (filters applied, result count, etc.)
                - synthesized_answer: LLM-generated answer (if custom_prompt provided)

        Raises:
            SessionNotSetError: If no user session is active
        """
        if user_id:
            original_user = self._current_user
            self._current_user = user_id
            try:
                return self.advanced_query(
                    query, filters, date_range, relevance_threshold,
                    max_results, custom_prompt, require_all_filters,
                    sort_by, include_metadata
                )
            finally:
                self._current_user = original_user

        self._ensure_session_active()

        # Build comprehensive filters (without user_id in metadata)
        query_filters = {}
        if filters:
            query_filters.update(filters)

        # Add date range filtering
        if date_range:
            start_date, end_date = date_range
            query_filters["date_range"] = {"start": start_date, "end": end_date}

        # Perform initial search with higher limit for filtering
        initial_limit = max_results * 3 if filters else max_results

        try:
            # Pass user_id as parameter, not in filters
            raw_results = self.memory.search(
                query,
                user_id=self._current_user,
                limit=initial_limit
            )
        except Exception as e:
            return {
                "results": [],
                "metadata": {
                    "error": str(e),
                    "query": query,
                    "filters_applied": query_filters,
                    "user_id": self._current_user
                }
            }

        # Post-process results with advanced filtering
        filtered_results = self._apply_advanced_filters(
            results=raw_results or [],
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
                "sort_by": sort_by,
                "user_id": self._current_user
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
            # Generate response using mem0's search with the formatted prompt
            response = self.memory.search(formatted_prompt, limit=1)
            if response and len(response) > 0:
                return response[0].get("text", "No synthesis available")
            return "Unable to generate synthesis"
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"

    # ==================== Utility Methods ====================

    def save(self) -> None:
        """Explicitly save memory to disk."""
        try:
            if hasattr(self.memory, 'vector_store'):
                self.memory.vector_store.save()
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")

    def reset_index(self) -> None:
        """
        Manually reset the FAISS index. Use this if you encounter dimension mismatch errors.
        WARNING: This will delete all existing memories!
        """
        if not self.persist_dir or not self.collection_name:
            print("No persist directory or collection name set")
            return

        persist_path = Path(self.persist_dir)
        if persist_path.exists():
            index_file = persist_path / f"{self.collection_name}.index"
            pkl_file = persist_path / f"{self.collection_name}.pkl"

            deleted = []
            if index_file.exists():
                index_file.unlink()
                deleted.append("index")
            if pkl_file.exists():
                pkl_file.unlink()
                deleted.append("pkl")

            if deleted:
                print(f"✓ Deleted {', '.join(deleted)} files. Please rebuild the memory instance.")
            else:
                print("No index files found to delete.")
        """Get statistics about the memory store."""
        stats = {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "current_user": self._current_user,
            "active_sessions": len(self._active_sessions),
            "session_users": list(self._active_sessions.keys()),
            "reranking_enabled": self.enable_reranking,
            "reranking_model": self.re_ranking_model
        }

        if self.persist_dir and self.collection_name:
            index_path = Path(self.persist_dir) / f"{self.collection_name}.index"
            if index_path.exists():
                stats["index_size_mb"] = index_path.stat().st_size / (1024 * 1024)
                stats["last_modified"] = datetime.fromtimestamp(
                    index_path.stat().st_mtime
                ).isoformat()

        return stats

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory (with user session check).

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted successfully

        Raises:
            SessionNotSetError: If no user session is active
        """
        self._ensure_session_active()

        try:
            self.memory.delete(memory_id)
            self.save()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def clear_user_memories(self, user_id: Optional[str] = None) -> int:
        """
        Clear all memories for a specific user.

        Args:
            user_id: User to clear memories for (defaults to current user)

        Returns:
            Number of memories cleared

        Raises:
            SessionNotSetError: If no user session is active
        """
        target_user = user_id if user_id else self._current_user

        if not target_user:
            raise SessionNotSetError(
                "No active user session. Call set_user_session() first."
            )

        try:
            # Get all memories for the user
            all_memories = self.get_all(user_id=target_user)
            count = 0

            for mem in all_memories:
                if mem.get("id"):
                    if self.memory.delete(mem["id"]):
                        count += 1

            self.save()
            return count
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return 0
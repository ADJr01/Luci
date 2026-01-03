from mem0 import Memory

def mem_ollama(
    collection_name: str,
    persist_dir: str,
    ollama_model: str,
    embedding_model: str,
    re_ranking_model: str,
    max_tokens: int = 3000,
    temperature: float = 0.0,
    distance_metric: str = "cosine",
    ollama_base_url: str = "http://localhost:11434/"
):
    return {
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": collection_name,
                "path": persist_dir,
                "distance_strategy": distance_metric,
                "index_type": "HNSW",
                "index_params": {
                    "M": 32,
                    "ef_construction": 200
                },
                "enable_metadata_filter": True
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": ollama_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "ollama_base_url": ollama_base_url
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embedding_model,
                "ollama_base_url": ollama_base_url,
                "normalize_embeddings": True
            }
        },
        "reranker": {
            "provider": "ollama",
            "config": {
                "model": re_ranking_model,
                "ollama_base_url": ollama_base_url
            }
        }
    }
